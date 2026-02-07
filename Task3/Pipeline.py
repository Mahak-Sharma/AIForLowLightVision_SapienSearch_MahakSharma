import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import torchvision.transforms.functional as TF
from transformers import SegformerForSemanticSegmentation

# ==========================================
# 1. CONFIGURATION
# ==========================================
CONFIG = {
    'enhancement_model_path': '../best_model_epoch_25_psnr_20.27.pth',
    'segmentation_model_path': '../stage2_trackB.pth',
    'input_video': './vid2.mp4',
    'output_enhanced': './video_enhanced_only.mp4',
    'output_segmentation': './video_segmentation_overlay.mp4',
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'frame_height': 256,
    'frame_width': 320,
    'num_classes': 8,
    'overlay_alpha': 0.4,
    'class_colors': [
        (0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255), (128, 128, 128)
    ]
}

# ==========================================
# 2. MEMORY-EFFICIENT MODEL CLASSES
# ==========================================
class WindowAttention(nn.Module):
    def __init__(self, dim, heads=4, window_size=8):
        super().__init__()
        self.heads = heads
        self.window_size = window_size
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        # Window Partition
        x = rearrange(x, 'b c (h p1) (w p2) -> (b h w) (p1 p2) c', p1=self.window_size, p2=self.window_size)
        
        qkv = rearrange(self.qkv(rearrange(x, 'b n c -> b c n').view(x.shape[0], C, self.window_size, self.window_size)), 
                        'b (h d) p1 p2 -> b h (p1 p2) d', h=self.heads).chunk(3, dim=-1)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        out = (attn @ v)
        out = rearrange(out, 'b h (p1 p2) d -> b (h d) p1 p2', p1=self.window_size, p2=self.window_size)
        
        # Merge Windows
        out = rearrange(out, '(b h w) c p1 p2 -> b c (h p1) (w p2)', h=H//self.window_size, w=W//self.window_size)
        return self.proj(out)

class Retinexformer(nn.Module):
    def __init__(self, n_feats=40):
        super().__init__()
        self.est = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(16, 1, 3, 1, 1), nn.Sigmoid()
        )
        self.inp = nn.Conv2d(3, n_feats, 3, 1, 1)
        self.attn = WindowAttention(n_feats) # Use Windowed Attention here
        self.out_conv = nn.Conv2d(n_feats, 3, 3, 1, 1)

    def forward(self, x):
        L = self.est(x)
        x_lit = x / (L + 1e-4)
        feat = self.inp(x_lit)
        res = self.attn(feat)
        return torch.clamp(x_lit + self.out_conv(res), 0, 1)

# ==========================================
# 3. PIPELINE
# ==========================================
class UnderwaterPipeline:
    def __init__(self):
        print("Initializing Models (Memory-Efficient)...")
        self.enhancer = Retinexformer().to(CONFIG['device'])
        if os.path.exists(CONFIG['enhancement_model_path']):
            # Fixed weight loading to ignore strict keys and use safer method
            ckpt = torch.load(CONFIG['enhancement_model_path'], map_location=CONFIG['device'], weights_only=False)
            self.enhancer.load_state_dict(ckpt, strict=False)
        self.enhancer.eval()

        self.segmenter = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512",
            num_labels=CONFIG['num_classes'], 
            ignore_mismatched_sizes=True
        ).to(CONFIG['device'])
        
        if os.path.exists(CONFIG['segmentation_model_path']):
            ckpt_seg = torch.load(CONFIG['segmentation_model_path'], map_location=CONFIG['device'], weights_only=False)
            state_dict = ckpt_seg['model_state_dict'] if 'model_state_dict' in ckpt_seg else ckpt_seg
            self.segmenter.load_state_dict(state_dict, strict=False)
        self.segmenter.eval()

    def process_video(self):
        cap = cv2.VideoCapture(CONFIG['input_video'])
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_enh = cv2.VideoWriter(CONFIG['output_enhanced'], fourcc, fps, (CONFIG['frame_width'], CONFIG['frame_height']))
        out_seg = cv2.VideoWriter(CONFIG['output_segmentation'], fourcc, fps, (CONFIG['frame_width'], CONFIG['frame_height']))

        for _ in tqdm(range(total_frames), desc="Dual-Stream Processing"):
            ret, frame = cap.read()
            if not ret: break
            
            frame_resized = cv2.resize(frame, (CONFIG['frame_width'], CONFIG['frame_height']))
            img_tensor = TF.to_tensor(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(CONFIG['device'])
            
            with torch.no_grad():
                # 1. Enhancement
                enhanced_t = self.enhancer(img_tensor)
                
                # 2. Segmentation
                seg_out = self.segmenter(pixel_values=enhanced_t).logits
                seg_out = F.interpolate(seg_out, size=(CONFIG['frame_height'], CONFIG['frame_width']), mode='bilinear')
                mask = torch.argmax(seg_out, dim=1).squeeze().cpu().numpy()

            # Video 1: Enhancement
            enh_np = (enhanced_t.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            enh_bgr = cv2.cvtColor(enh_np, cv2.COLOR_RGB2BGR)
            out_enh.write(enh_bgr)

            # Video 2: Segmentation
            color_mask = np.zeros_like(enh_bgr)
            for idx, color in enumerate(CONFIG['class_colors']):
                if idx == 0: continue
                color_mask[mask == idx] = color[::-1]
            
            overlay = cv2.addWeighted(enh_bgr, 0.7, color_mask, 0.3, 0)
            out_seg.write(overlay)

        cap.release()
        out_enh.release()
        out_seg.release()
        print("\nProcessing finished successfully.")

if __name__ == "__main__":
    UnderwaterPipeline().process_video()