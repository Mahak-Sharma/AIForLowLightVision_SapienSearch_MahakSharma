"""
Underwater Semantic Segmentation using SegFormer
Dataset: AI Summit Track B - Underwater Imagery
Resolution: 320x256 for consistency
Split: 80% train, 20% validation
"""

import os
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import SegformerConfig, SegformerForSemanticSegmentation
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Dataset paths
DATASET_ROOT = "./rclone-v1.73.0-linux-amd64/TrackB/dataset"
IMG_DIR = os.path.join(DATASET_ROOT, "images")
MASK_DIR = os.path.join(DATASET_ROOT, "masks/combined")

# Model & training config
IMG_HEIGHT = 256
IMG_WIDTH = 320
NUM_CLASSES = 8
BATCH_SIZE = 4
EPOCHS = 30
LEARNING_RATE = 6e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Best model save path
BEST_MODEL_PATH = "./best_segformer_underwater.pth"

print(f"Device: {DEVICE}")
print(f"Images: {IMG_DIR}")
print(f"Masks: {MASK_DIR}")

# ==============================================================================
# COLOR MAPPING (8 classes)
# ==============================================================================

COLOR_MAP = {
    (0, 0, 0): 0,           # Background/Sea-floor
    (255, 0, 0): 1,         # Fish
    (0, 255, 0): 2,         # Reefs
    (0, 0, 255): 3,         # Plants
    (255, 255, 0): 4,       # Wrecks
    (255, 0, 255): 5,       # Divers
    (0, 255, 255): 6,       # Robots
    (128, 128, 128): 7      # Others
}

IDX_TO_COLOR = {v: k for k, v in COLOR_MAP.items()}


def rgb_to_label(mask):
    """Convert RGB mask to integer label mask"""
    mask = np.array(mask)
    label = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int64)

    for rgb, idx in COLOR_MAP.items():
        matches = np.all(mask == np.array(rgb), axis=-1)
        label[matches] = idx

    return label


# ==============================================================================
# DATASET CLASS
# ==============================================================================

class UnderwaterSegDataset(Dataset):
    def __init__(self, img_dir, mask_dir, pairs):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_name, mask_name = self.pairs[idx]

        # Load image and mask
        img = Image.open(os.path.join(self.img_dir, img_name)).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, mask_name)).convert("RGB")

        # Resize to consistent resolution (320x256)
        img = img.resize((IMG_WIDTH, IMG_HEIGHT), Image.BILINEAR)
        mask = mask.resize((IMG_WIDTH, IMG_HEIGHT), Image.NEAREST)

        # Convert to tensors
        img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(rgb_to_label(mask)).long()

        return img, mask


# ==============================================================================
# BUILD IMAGE-MASK PAIRS
# ==============================================================================

def build_pairs(img_dir, mask_dir):
    """Match images with their corresponding masks"""
    imgs = sorted([f for f in os.listdir(img_dir) 
                   if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    masks = set(os.listdir(mask_dir))

    pairs = []
    for img in imgs:
        base = os.path.splitext(img)[0]
        # Try different extensions for masks
        for ext in [".bmp", ".png", ".jpg"]:
            mask_name = base + ext
            if mask_name in masks:
                pairs.append((img, mask_name))
                break

    print(f"Found {len(pairs)} image-mask pairs")
    return pairs


# ==============================================================================
# LOAD AND SPLIT DATA (80:20)
# ==============================================================================

all_pairs = build_pairs(IMG_DIR, MASK_DIR)

train_pairs, val_pairs = train_test_split(
    all_pairs, 
    test_size=0.2, 
    random_state=42,
    shuffle=True
)

print(f"Train samples: {len(train_pairs)}")
print(f"Val samples: {len(val_pairs)}")

train_ds = UnderwaterSegDataset(IMG_DIR, MASK_DIR, train_pairs)
val_ds = UnderwaterSegDataset(IMG_DIR, MASK_DIR, val_pairs)

train_loader = DataLoader(
    train_ds, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=2,
    pin_memory=True
)
val_loader = DataLoader(
    val_ds, 
    batch_size=2, 
    shuffle=False, 
    num_workers=2,
    pin_memory=True
)


# ==============================================================================
# MODEL SETUP
# ==============================================================================

config = SegformerConfig.from_pretrained(
    "nvidia/segformer-b2-finetuned-ade-512-512",
    num_labels=NUM_CLASSES
)

model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b2-finetuned-ade-512-512",
    config=config,
    ignore_mismatched_sizes=True,
    use_safetensors=True  # Use safetensors to avoid torch.load vulnerability
).to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

print("Model loaded: SegFormer-B2")


# ==============================================================================
# METRICS
# ==============================================================================

def mean_iou(pred, target, num_classes=NUM_CLASSES):
    """Calculate mean IoU across all classes"""
    ious = []
    for c in range(num_classes):
        p = pred == c
        t = target == c
        inter = (p & t).sum().item()
        union = (p | t).sum().item()
        if union == 0:
            ious.append(1.0)  # Perfect if class not present
        else:
            ious.append(inter / union)
    return np.mean(ious)


def mean_f1(pred, target):
    """Calculate macro-averaged F1 score"""
    return f1_score(
        target.cpu().numpy().flatten(),
        pred.cpu().numpy().flatten(),
        average="macro",
        zero_division=1
    )


# ==============================================================================
# TRAINING FUNCTION
# ==============================================================================

def train_one_epoch(loader):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    pbar = tqdm(loader, desc="Training")
    for imgs, masks in pbar:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(pixel_values=imgs, labels=masks)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / len(loader)


# ==============================================================================
# EVALUATION FUNCTION
# ==============================================================================

def evaluate(loader):
    """Evaluate model on validation set"""
    model.eval()
    miou_list, f1_list = [], []

    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc="Evaluating"):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

            # Get predictions
            outputs = model(pixel_values=imgs)
            logits = outputs.logits
            
            # Upsample logits to match mask resolution
            logits = F.interpolate(
                logits,
                size=(IMG_HEIGHT, IMG_WIDTH),
                mode="bilinear",
                align_corners=False
            )

            preds = torch.argmax(logits, dim=1)

            # Calculate metrics for this batch
            miou_list.append(mean_iou(preds, masks))
            f1_list.append(mean_f1(preds, masks))

    return np.mean(miou_list), np.mean(f1_list)


# ==============================================================================
# TRAINING LOOP WITH BEST MODEL SAVING
# ==============================================================================

best_miou = 0.0

print("\n" + "="*60)
print("TRAINING START")
print("="*60 + "\n")

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print("-" * 40)
    
    # Train
    train_loss = train_one_epoch(train_loader)
    
    # Validate
    val_miou, val_f1 = evaluate(val_loader)
    
    # Print results
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val mIoU:   {val_miou:.4f}")
    print(f"Val F1:     {val_f1:.4f}")
    
    # Save best model
    if val_miou > best_miou:
        best_miou = val_miou
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'miou': val_miou,
            'f1': val_f1,
        }, BEST_MODEL_PATH)
        print(f"✓ Best model saved! (mIoU: {val_miou:.4f})")

print("\n" + "="*60)
print("TRAINING COMPLETE")
print(f"Best Validation mIoU: {best_miou:.4f}")
print(f"Best model saved at: {BEST_MODEL_PATH}")
print("="*60)


# ==============================================================================
# INFERENCE FUNCTION (for later use)
# ==============================================================================

def predict_image(image_path, model, save_path=None):
    """
    Predict segmentation mask for a single image
    
    Args:
        image_path: Path to input image
        model: Trained SegFormer model
        save_path: Optional path to save colored mask
    
    Returns:
        Predicted mask (numpy array)
    """
    model.eval()
    
    # Load and preprocess image
    img = Image.open(image_path).convert("RGB")
    img_resized = img.resize((IMG_WIDTH, IMG_HEIGHT), Image.BILINEAR)
    img_tensor = torch.from_numpy(np.array(img_resized)).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(DEVICE)
    
    # Predict
    with torch.no_grad():
        outputs = model(pixel_values=img_tensor)
        logits = outputs.logits
        
        logits = F.interpolate(
            logits,
            size=(IMG_HEIGHT, IMG_WIDTH),
            mode="bilinear",
            align_corners=False
        )
        
        pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
    
    # Convert to color mask
    color_mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
    for cls, color in IDX_TO_COLOR.items():
        color_mask[pred == cls] = color
    
    # Save if requested
    if save_path:
        mask_img = Image.fromarray(color_mask)
        mask_img.save(save_path)
        print(f"Mask saved to: {save_path}")
    
    return pred


# ==============================================================================
# LOAD BEST MODEL (optional - for inference later)
# ==============================================================================

def load_best_model():
    """Load the best saved model"""
    checkpoint = torch.load(BEST_MODEL_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch']}")
    print(f"mIoU: {checkpoint['miou']:.4f}, F1: {checkpoint['f1']:.4f}")
    return model


# Example usage after training:
# best_model = load_best_model()
# predict_image("path/to/test/image.jpg", best_model, "output_mask.bmp")