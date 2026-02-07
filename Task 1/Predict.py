"""
Retinexformer Image Enhancement - Prediction Script (Conservative Memory Version)
Always uses tiling to avoid OOM errors
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from einops import rearrange
from tqdm import tqdm
import zipfile

# ==========================================
# CONFIGURATION
# ==========================================
TEST_IMG_DIR = "./rclone-v1.73.0-linux-amd64/dataset/IMAGE_ENHANCEMENT_Test_zipped"
OUTPUT_DIR = "./enhanced_predictions"
MODEL_PATH = "best_model_epoch_25_psnr_20.27.pth"
TEAM_NAME = "SapienSearch"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TILE_SIZE = 512  # Process in 512x512 tiles
OVERLAP = 64     # Overlap for smooth blending

print(f"Device: {DEVICE}")
print(f"Model: {MODEL_PATH}")
print(f"Team Name: {TEAM_NAME}")
print(f"Tile Size: {TILE_SIZE}x{TILE_SIZE} (overlap: {OVERLAP}px)")

# ==========================================
# MODEL ARCHITECTURE
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
    def forward(self, x): 
        return self.net(x)


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
# LOAD MODEL
# ==========================================
print("\nLoading model...")
model = Retinexformer().to(DEVICE)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
model.load_state_dict(checkpoint)
model.eval()
print("✓ Model loaded successfully")


# ==========================================
# PREDICTION FUNCTION (ALWAYS USES TILING)
# ==========================================
def predict_image_tiled(image_path, model, tile_size=TILE_SIZE, overlap=OVERLAP):
    """
    Process image using tiling - ALWAYS, regardless of size
    This ensures no OOM errors even on large images
    """
    # Load image
    img = Image.open(image_path).convert('RGB')
    
    # Convert to numpy for tiling
    img_np = np.array(img).astype(np.float32) / 255.0
    h, w, c = img_np.shape
    
    # Create output array
    output = np.zeros_like(img_np)
    weight_map = np.zeros((h, w), dtype=np.float32)
    
    # Calculate number of tiles
    stride = tile_size - overlap
    n_tiles_h = max(1, (h - overlap + stride - 1) // stride)
    n_tiles_w = max(1, (w - overlap + stride - 1) // stride)
    
    # Process tiles
    for i in range(n_tiles_h):
        for j in range(n_tiles_w):
            # Calculate tile coordinates
            y_start = i * stride
            x_start = j * stride
            y_end = min(y_start + tile_size, h)
            x_end = min(x_start + tile_size, w)
            
            # Adjust start if we're at the edge
            if y_end == h and y_end - y_start < tile_size:
                y_start = max(0, h - tile_size)
            if x_end == w and x_end - x_start < tile_size:
                x_start = max(0, w - tile_size)
            
            # Extract tile
            tile = img_np[y_start:y_end, x_start:x_end, :]
            
            # Convert to tensor
            tile_tensor = torch.from_numpy(tile).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
            
            # Process tile
            with torch.no_grad():
                enhanced_tile = model(tile_tensor)
            
            # Convert back to numpy
            enhanced_tile = enhanced_tile.squeeze(0).permute(1, 2, 0).cpu().numpy()
            
            # Free GPU memory immediately
            del tile_tensor
            torch.cuda.empty_cache()
            
            # Create weight for blending
            tile_h, tile_w = enhanced_tile.shape[:2]
            weight_tile = np.ones((tile_h, tile_w), dtype=np.float32)
            
            # Apply feathering at edges for smooth blending
            if overlap > 0:
                fade = min(overlap, 32)
                for k in range(fade):
                    alpha = k / fade
                    # Top edge
                    if y_start > 0 and k < tile_h:
                        weight_tile[k, :] *= alpha
                    # Bottom edge
                    if y_end < h and k < tile_h:
                        weight_tile[-(k+1), :] *= alpha
                    # Left edge
                    if x_start > 0 and k < tile_w:
                        weight_tile[:, k] *= alpha
                    # Right edge
                    if x_end < w and k < tile_w:
                        weight_tile[:, -(k+1)] *= alpha
            
            # Add to output with blending
            output[y_start:y_end, x_start:x_end, :] += enhanced_tile * weight_tile[:, :, np.newaxis]
            weight_map[y_start:y_end, x_start:x_end] += weight_tile
    
    # Normalize by weight map
    output = output / np.maximum(weight_map[:, :, np.newaxis], 1e-8)
    
    # Convert back to PIL Image
    output = np.clip(output * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(output)


# ==========================================
# PROCESS TEST IMAGES
# ==========================================
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Get all test images
test_images = sorted([f for f in os.listdir(TEST_IMG_DIR) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

print(f"\nFound {len(test_images)} test images")
print("Generating enhanced images (using tiling for all images)...\n")

# Process each image
failed_images = []
for idx, img_name in enumerate(tqdm(test_images, desc="Processing")):
    img_path = os.path.join(TEST_IMG_DIR, img_name)
    
    try:
        # Generate enhanced image
        enhanced_img = predict_image_tiled(img_path, model)
        
        # Save
        base_name = os.path.splitext(img_name)[0]
        output_path = os.path.join(OUTPUT_DIR, f"{base_name}.png")
        enhanced_img.save(output_path, format='PNG')
        
    except Exception as e:
        print(f"\nError processing {img_name}: {e}")
        failed_images.append(img_name)
    
    finally:
        # Clear GPU cache after EVERY image
        torch.cuda.empty_cache()

print(f"\n✓ Processing complete!")
print(f"  Successfully processed: {len(test_images) - len(failed_images)}/{len(test_images)}")
if failed_images:
    print(f"  Failed images: {failed_images}")
print(f"  Output directory: {OUTPUT_DIR}")


# ==========================================
# CREATE SUBMISSION ZIP
# ==========================================
zip_path = f"./{TEAM_NAME}.zip"

with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    png_files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.png')])
    
    for idx, filename in enumerate(png_files):
        file_path = os.path.join(OUTPUT_DIR, filename)
        # Rename to img_XXXX.png format
        new_name = f"img_{idx:04d}.png"
        zipf.write(file_path, arcname=f"{TEAM_NAME}/{new_name}")

print(f"\n✓ Submission zip created: {zip_path}")
print(f"   Structure: {TEAM_NAME}.zip/{TEAM_NAME}/img_0000.png to img_{len(png_files)-1:04d}.png")
print(f"   Total images in zip: {len(png_files)}")