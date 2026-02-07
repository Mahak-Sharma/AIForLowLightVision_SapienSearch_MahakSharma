import os
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from torchvision.transforms import RandomCrop
from torch.amp import autocast, GradScaler
from PIL import Image
from einops import rearrange
from piq import ssim, LPIPS
import time

# ==========================================
# 1. CONFIG & PATHS
# ==========================================
INPUT_DIR = "./rclone-v1.73.0-linux-amd64/dataset/inputs"
TARGET_DIR = "./rclone-v1.73.0-linux-amd64/dataset/targets"
PRETRAINED_PATH = "./rclone-v1.73.0-linux-amd64/retinexWeight/LOL_v1.pth"
SAVE_PATH = "best_finetune_model.pth"

CONFIG = {
    'batch_size': 4,
    'img_size': 256,
    'lr': 5e-5,
    'epochs': 25,
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

# ==========================================
# 2. METRICS
# ==========================================
def calculate_uciqe(img_tensor):
    img = img_tensor.detach().cpu().numpy().squeeze().transpose(1, 2, 0)
    img = (img * 255).astype(np.uint8)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    sigma_c = np.sqrt(np.var(a) + np.var(b))
    con_l = np.percentile(l, 99) - np.percentile(l, 1)
    mu_s = np.mean(np.sqrt(a**2 + b**2))
    return 0.4680 * sigma_c + 0.2745 * con_l + 0.2576 * mu_s

def calculate_psnr(img1, img2):
    mse = F.mse_loss(img1, img2)
    return 20 * torch.log10(1.0 / torch.sqrt(mse)) if mse > 0 else torch.tensor(100.0)

# ==========================================
# 3. CORE ARCHITECTURE
# ==========================================
class Illumination_Estimator(nn.Module):
    def __init__(self, n_feats=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, n_feats, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(n_feats, 1, 3, 1, 1),
            nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

class IG_MSA(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim // heads
        self.scale = self.dim_head ** -0.5
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.project = nn.Conv2d(dim, dim, 1)

    def forward(self, x, L):
        B, C, H, W = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=1)
        q = rearrange(q, 'b (h d) y x -> b h (y x) d', h=self.num_heads)
        k = rearrange(k, 'b (h d) y x -> b h (y x) d', h=self.num_heads)
        v = rearrange(v, 'b (h d) y x -> b h (y x) d', h=self.num_heads)
        L_map = F.interpolate(L, size=(H, W), mode='bilinear', align_corners=False)
        v = v * rearrange(L_map, 'b c y x -> b 1 (y x) c')
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b h (y x) d -> b (h d) y x', y=H, x=W)
        return self.project(out)

class Retinexformer(nn.Module):
    def __init__(self, n_feats=40):
        super().__init__()
        self.est = Illumination_Estimator()
        self.inp = nn.Conv2d(3, n_feats, 3, 1, 1)
        self.igt = IG_MSA(n_feats)
        self.out = nn.Conv2d(n_feats, 3, 3, 1, 1)

    def forward(self, x):
        L = self.est(x)
        x_lit = x / (L + 1e-4)
        feat = self.inp(x_lit)
        feat_small = F.interpolate(feat, scale_factor=0.25, mode='bilinear')
        L_small = F.interpolate(L, scale_factor=0.25, mode='bilinear')
        res_small = self.igt(feat_small, L_small)
        res = F.interpolate(res_small, size=(x.shape[2], x.shape[3]), mode='bilinear')
        return torch.clamp(x_lit + self.out(res), 0, 1)

# ==========================================
# 4. DATASET
# ==========================================
class PairedUnderwaterDataset(Dataset):
    def __init__(self, input_dir, target_dir, img_size=256, is_train=True):
        self.input_dir, self.target_dir = input_dir, target_dir
        self.img_size, self.is_train = img_size, is_train
        self.input_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    def __len__(self): return len(self.input_files)

    def __getitem__(self, idx):
        fn = self.input_files[idx]
        img_in = Image.open(os.path.join(self.input_dir, fn)).convert('RGB')
        img_tgt = Image.open(os.path.join(self.target_dir, fn)).convert('RGB')
        w, h = img_in.size
        if h < self.img_size or w < self.img_size:
            img_in = TF.resize(img_in, [max(h, self.img_size), max(w, self.img_size)])
            img_tgt = TF.resize(img_tgt, [max(h, self.img_size), max(w, self.img_size)])
        if self.is_train:
            i, j, h, w = RandomCrop.get_params(img_in, output_size=(self.img_size, self.img_size))
            img_in, img_tgt = TF.crop(img_in, i, j, h, w), TF.crop(img_tgt, i, j, h, w)
            if random.random() > 0.5: img_in, img_tgt = TF.hflip(img_in), TF.hflip(img_tgt)
        else:
            img_in, img_tgt = TF.resize(img_in, [self.img_size, self.img_size]), TF.resize(img_tgt, [self.img_size, self.img_size])
        return TF.to_tensor(img_in), TF.to_tensor(img_tgt)

# ==========================================
# 5. TRAINING LOOP
# ==========================================
def train():
    model = Retinexformer().to(CONFIG['device'])
    best_psnr = -1.0
    
    if os.path.exists(PRETRAINED_PATH):
        ckpt = torch.load(PRETRAINED_PATH, map_location=CONFIG['device'], weights_only=True)
        model.load_state_dict(ckpt.get('params', ckpt), strict=False)
        for name, param in model.named_parameters():
            if "est." in name: param.requires_grad = False
        print("Success: Pretrained weights loaded.")

    train_ds = PairedUnderwaterDataset(INPUT_DIR, TARGET_DIR, CONFIG['img_size'], True)
    val_ds = PairedUnderwaterDataset(INPUT_DIR, TARGET_DIR, CONFIG['img_size'], False)
    train_loader = DataLoader(train_ds, CONFIG['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, 1, shuffle=False)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=CONFIG['lr'])
    scaler = torch.amp.GradScaler('cuda')
    criterion = nn.L1Loss()
    lpips_fn = LPIPS(replace_pooling=False).to(CONFIG['device'])

    for epoch in range(CONFIG['epochs']):
        model.train()
        total_loss = 0
        start_time = time.time()
        
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(CONFIG['device']), y.to(CONFIG['device'])
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                pred = model(x)
                loss = criterion(pred, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            if i % 100 == 0:
                print(f"Epoch {epoch+1} | Batch {i}/{len(train_loader)} | Loss: {loss.item():.4f}")

        # --- VALIDATION ---
        model.eval()
        p_list, s_list, u_list, l_list = [], [], [], []
        print(f"Starting Validation for Epoch {epoch+1}...")
        
        with torch.no_grad():
            for idx, (x, y) in enumerate(val_loader):
                x, y = x.to(CONFIG['device']), y.to(CONFIG['device'])
                pred = model(x)
                
                # We calculate metrics every 10 images to speed up validation
                # Since you have 5600 validation images, doing 560 takes less time
                if idx % 10 == 0:
                    p_list.append(calculate_psnr(pred, y).item())
                    s_list.append(ssim(pred, y, data_range=1.0).item())
                    l_list.append(lpips_fn(pred, y).item())
                    u_list.append(calculate_uciqe(pred))
                
                if idx % 500 == 0:
                    print(f"Validated {idx}/{len(val_loader)} images...")

        current_psnr = np.mean(p_list)
        print(f"\n==> Epoch {epoch+1} Summary:")
        print(f"PSNR: {current_psnr:.2f} | SSIM: {np.mean(s_list):.4f} | LPIPS: {np.mean(l_list):.4f} | UCIQE: {np.mean(u_list):.4f}")

        # --- SAVE BEST MODEL ---
        if current_psnr > best_psnr:
            best_psnr = current_psnr
            epoch_save_path = f"best_model_epoch_{epoch+1}_psnr_{current_psnr:.2f}.pth"
            torch.save(model.state_dict(), epoch_save_path)
            tmp_path = SAVE_PATH + ".tmp"
            torch.save(model.state_dict(), tmp_path)
            os.rename(tmp_path, SAVE_PATH)
            print(f" New Best PSNR! Model saved to {SAVE_PATH} and {epoch_save_path}")
        
        if epoch == 19:
            for p in model.parameters(): p.requires_grad = True
            print(" Unfrozen full model for final fine-tuning.")

if __name__ == "__main__":
    train()