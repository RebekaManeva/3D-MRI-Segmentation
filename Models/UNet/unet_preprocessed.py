import torch
import os, glob, random, time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
from medpy.metric.binary import hd95
import shutil
import re
import torch.optim as optim
from scipy import ndimage
from scipy.ndimage import zoom

DATA_ROOT = "#"
TRAIN_DIR = os.path.join(DATA_ROOT, "Train")
VAL_DIR = os.path.join(DATA_ROOT, "Validation")
TEST_DIR = os.path.join(DATA_ROOT, "Test")
PLOTS_DIR = "#"
model_dir = "#"

TARGET_SHAPE = (128, 128, 128)

BATCH_SIZE = 2
NUM_WORKERS = 2
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100


class BratsDataset(Dataset):
    def __init__(self, root_dir, target_shape=(128, 128, 128), normalization=True):
        super().__init__()
        self.root_dir = root_dir
        self.target_shape = target_shape
        self.normalization = normalization

        self.patient_folders = []
        for d in sorted(os.listdir(root_dir)):
            patient_path = os.path.join(root_dir, d)
            if not os.path.isdir(patient_path):
                continue

            #volume_path = os.path.join(patient_path, "multi_channel_volume.npy")
            volume_path = os.path.join(patient_path, "data.npy")
            mask_path = os.path.join(patient_path, "mask.npy")

            if os.path.exists(volume_path) and os.path.exists(mask_path):
                self.patient_folders.append(d)
            else:
                print(f"Missing files {d}")

        print(f"Found {len(self.patient_folders)} patients in {root_dir}")

    def __len__(self):
        return len(self.patient_folders)

    def resize_volume(self, volume, target_shape):
        current_shape = volume.shape[:3]
        zoom_factors = [
            target_shape[0] / current_shape[0],
            target_shape[1] / current_shape[1],
            target_shape[2] / current_shape[2]
        ]
        if volume.ndim == 4:
            zoom_factors.append(1)
        resized = zoom(volume, zoom_factors, order=1)
        return resized

    def resize_mask(self, mask, target_shape):
        current_shape = mask.shape
        zoom_factors = [
            target_shape[0] / current_shape[0],
            target_shape[1] / current_shape[1],
            target_shape[2] / current_shape[2]
        ]
        resized = zoom(mask, zoom_factors, order=0)
        return resized

    def normalize_channel(self, channel):
        mask = channel > 0
        if mask.sum() > 0:
            mean = channel[mask].mean()
            std = channel[mask].std()
            if std > 0:
                channel[mask] = (channel[mask] - mean) / std
        return channel

    def mask_to_brats_regions(self, mask):
        mask_regions = np.zeros((*mask.shape, 3), dtype=np.float32)
        mask_regions[..., 0] = (mask == 4).astype(np.float32)
        mask_regions[..., 1] = ((mask == 1) | (mask == 4)).astype(np.float32)
        mask_regions[..., 2] = ((mask == 1) | (mask == 2) | (mask == 4)).astype(np.float32)
        return mask_regions

    def __getitem__(self, idx):
        patient_folder = os.path.join(self.root_dir, self.patient_folders[idx])

        # image = np.load(os.path.join(patient_folder, "multi_channel_volume.npy"))
        try:
            image = np.load(os.path.join(patient_folder, "data.npy"))
            mask = np.load(os.path.join(patient_folder, "mask.npy"))

            image_resized = self.resize_volume(image, self.target_shape)
            mask_resized = self.resize_mask(mask, self.target_shape)

            if self.normalization:
                for c in range(image_resized.shape[-1]):
                    image_resized[..., c] = self.normalize_channel(image_resized[..., c])

            mask_regions = self.mask_to_brats_regions(mask_resized)

            image_tensor = torch.from_numpy(image_resized).permute(3, 0, 1, 2).float()
            mask_tensor = torch.from_numpy(mask_regions).permute(3, 0, 1, 2).float()

            return image_tensor, mask_tensor

        except Exception as e:
            print(f"Error loading: {self.patient_folders[idx]}: {e}")
            empty_image = torch.zeros(4, *self.target_shape, dtype=torch.float32)
            empty_mask = torch.zeros(3, *self.target_shape, dtype=torch.float32)
            return empty_image, empty_mask


class StandardConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class StandardUNet3D(nn.Module):

    def __init__(self, in_channels=4, out_channels=3):
        super().__init__()

        self.enc1 = StandardConvBlock(in_channels, 32)
        self.pool1 = nn.MaxPool3d(kernel_size=2)

        self.enc2 = StandardConvBlock(32, 64)
        self.pool2 = nn.MaxPool3d(kernel_size=2)

        self.enc3 = StandardConvBlock(64, 128)
        self.pool3 = nn.MaxPool3d(kernel_size=2)

        self.enc4 = StandardConvBlock(128, 256)
        self.pool4 = nn.MaxPool3d(kernel_size=2)

        self.bottleneck = StandardConvBlock(256, 512)

        self.up4 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.dec4 = StandardConvBlock(512, 256)  # 256 (up) + 256 (skip)

        self.up3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.dec3 = StandardConvBlock(256, 128)

        self.up2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec2 = StandardConvBlock(128, 64)

        self.up1 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec1 = StandardConvBlock(64, 32)

        self.out_conv = nn.Conv3d(32, out_channels, kernel_size=1)

    def forward(self, x):
        s1 = self.enc1(x)
        p1 = self.pool1(s1)

        s2 = self.enc2(p1)
        p2 = self.pool2(s2)

        s3 = self.enc3(p2)
        p3 = self.pool3(s3)

        s4 = self.enc4(p3)
        p4 = self.pool4(s4)

        b = self.bottleneck(p4)

        u4 = self.up4(b)
        u4 = torch.cat([u4, s4], dim=1)
        d4 = self.dec4(u4)

        u3 = self.up3(d4)
        u3 = torch.cat([u3, s3], dim=1)
        d3 = self.dec3(u3)

        u2 = self.up2(d3)
        u2 = torch.cat([u2, s2], dim=1)
        d2 = self.dec2(u2)

        u1 = self.up1(d2)
        u1 = torch.cat([u1, s1], dim=1)
        d1 = self.dec1(u1)

        return self.out_conv(d1)

class DiceBCELoss(nn.Module):
    def __init__(self, dice_weight=0.5, bce_weight=0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss()

    def dice_loss(self, pred, target, smooth=1e-6):
        pred = torch.sigmoid(pred)

        dice_per_class = []
        for c in range(pred.shape[1]):
            pred_c = pred[:, c].contiguous().view(-1)
            target_c = target[:, c].contiguous().view(-1)

            intersection = (pred_c * target_c).sum()
            dice = (2. * intersection + smooth) / (pred_c.sum() + target_c.sum() + smooth)
            dice_per_class.append(1 - dice)

        return torch.stack(dice_per_class).mean()

    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        bce = self.bce(pred, target)
        return self.dice_weight * dice + self.bce_weight * bce


def dice_score(pred, target, eps=1e-6):
    pred = (pred > 0.5).float()
    dice_per_class = []
    for c in range(pred.shape[1]):
        pred_c = pred[:, c]
        target_c = target[:, c]
        intersection = (pred_c * target_c).sum()
        dice = (2 * intersection + eps) / (pred_c.sum() + target_c.sum() + eps)
        dice_per_class.append(dice)
    return torch.stack(dice_per_class).mean()


def iou_score(pred, target, eps=1e-6):
    pred = (pred > 0.5).float()
    iou_per_class = []
    for c in range(pred.shape[1]):
        pred_c = pred[:, c]
        target_c = target[:, c]
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum() - intersection
        iou = (intersection + eps) / (union + eps)
        iou_per_class.append(iou)
    return torch.stack(iou_per_class).mean()


def precision_score(pred, target, eps=1e-6):
    pred = (pred > 0.5).float()
    prec_per_class = []
    for c in range(pred.shape[1]):
        pred_c = pred[:, c]
        target_c = target[:, c]
        tp = (pred_c * target_c).sum()
        fp = (pred_c * (1 - target_c)).sum()
        prec = (tp + eps) / (tp + fp + eps)
        prec_per_class.append(prec)
    return torch.stack(prec_per_class).mean()


def recall_score(pred, target, eps=1e-6):
    pred = (pred > 0.5).float()
    rec_per_class = []
    for c in range(pred.shape[1]):
        pred_c = pred[:, c]
        target_c = target[:, c]
        tp = (pred_c * target_c).sum()
        fn = ((1 - pred_c) * target_c).sum()
        rec = (tp + eps) / (tp + fn + eps)
        rec_per_class.append(rec)
    return torch.stack(rec_per_class).mean()


def f1_score(pred, target, eps=1e-6):
    p = precision_score(pred, target, eps)
    r = recall_score(pred, target, eps)
    return (2 * p * r + eps) / (p + r + eps)


def hd95_score(pred, target):
    pred = (pred > 0.5).cpu().numpy().astype(np.uint8)
    target = target.cpu().numpy().astype(np.uint8)

    if pred.ndim == 5:
        pred = pred[0, :].max(axis=0)
        target = target[0, :].max(axis=0)
    elif pred.ndim == 4:
        pred = pred[:].max(axis=0)
        target = target[:].max(axis=0)

    if pred.sum() == 0 and target.sum() == 0:
        return 0.0
    elif pred.sum() == 0 or target.sum() == 0:
        return np.nan

    try:
        return hd95(pred, target)
    except Exception as e:
        return np.nan


def plot_metrics(train_metrics, val_metrics, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(val_metrics['dice']) + 1)
    metric_names = ['dice', 'iou', 'precision', 'recall', 'f1', 'hd95']

    for metric_name in metric_names:
        if not val_metrics[metric_name] or not train_metrics[metric_name]:
            continue

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_metrics[metric_name],
                 label=f'Train {metric_name.upper()}',
                 marker='o', linestyle='-', markersize=4, linewidth=2)
        plt.plot(epochs, val_metrics[metric_name],
                 label=f'Validation {metric_name.upper()}',
                 marker='x', linestyle='--', markersize=4, linewidth=2)
        plt.title(f'Training and Validation {metric_name.upper()} Over Epochs', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)

        if metric_name == 'hd95':
            plt.ylabel('HD95 Distance (Lower is better)', fontsize=12)
        else:
            plt.ylabel(f'{metric_name.upper()} Score', fontsize=12)

        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = os.path.join(save_dir, f'{metric_name}_vs_epoch.png')
        plt.savefig(plot_path, dpi=150)
        plt.close()


def main_train():
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("torch.cuda.version:", torch.version.cuda)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("\nData loading...")
    train_ds = BratsDataset(TRAIN_DIR, target_shape=TARGET_SHAPE, normalization=True)
    val_ds = BratsDataset(VAL_DIR, target_shape=TARGET_SHAPE, normalization=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples: {len(val_ds)}")

    model = StandardUNet3D(in_channels=4, out_channels=3).to(device)
    criterion = DiceBCELoss(dice_weight=0.8, bce_weight=0.2)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    best_model_path = os.path.join(model_dir, "best_model.pth")
    last_model_path = os.path.join(model_dir, "last_model.pth")

    train_metrics = {'dice': [], 'iou': [], 'precision': [], 'recall': [], 'f1': [], 'hd95': []}
    val_metrics = {'dice': [], 'iou': [], 'precision': [], 'recall': [], 'f1': [], 'hd95': []}

    best_val_dice = 0.0
    early_stop_patience = 20
    patience_counter = 0

    print("Training started...")

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_dice, train_iou, train_prec, train_rec, train_f1, train_hd95 = [], [], [], [], [], []

        for batch_idx, (imgs, segs) in enumerate(train_loader):
            imgs, segs = imgs.to(device), segs.to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, segs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            with torch.no_grad():
                pred = torch.sigmoid(logits)

                for i in range(imgs.size(0)):
                    train_dice.append(dice_score(pred[i:i + 1], segs[i:i + 1]).item())
                    train_iou.append(iou_score(pred[i:i + 1], segs[i:i + 1]).item())
                    train_prec.append(precision_score(pred[i:i + 1], segs[i:i + 1]).item())
                    train_rec.append(recall_score(pred[i:i + 1], segs[i:i + 1]).item())
                    train_f1.append(f1_score(pred[i:i + 1], segs[i:i + 1]).item())
                    train_hd95.append(hd95_score(pred[i:i + 1], segs[i:i + 1]))

            if batch_idx % 10 == 0:
                print(
                    f"Epoch [{epoch + 1}/{NUM_EPOCHS}] Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")

        scheduler.step()

        train_metrics['dice'].append(np.nanmean(train_dice))
        train_metrics['iou'].append(np.nanmean(train_iou))
        train_metrics['precision'].append(np.nanmean(train_prec))
        train_metrics['recall'].append(np.nanmean(train_rec))
        train_metrics['f1'].append(np.nanmean(train_f1))
        train_metrics['hd95'].append(np.nanmean(train_hd95))

        model.eval()
        val_dice, val_iou, val_prec, val_rec, val_f1, val_hd95 = [], [], [], [], [], []
        with torch.no_grad():
            for imgs, segs in val_loader:
                imgs, segs = imgs.to(device), segs.to(device)
                logits = model(imgs)
                pred = torch.sigmoid(logits)

                for i in range(imgs.size(0)):
                    val_dice.append(dice_score(pred[i:i + 1], segs[i:i + 1]).item())
                    val_iou.append(iou_score(pred[i:i + 1], segs[i:i + 1]).item())
                    val_prec.append(precision_score(pred[i:i + 1], segs[i:i + 1]).item())
                    val_rec.append(recall_score(pred[i:i + 1], segs[i:i + 1]).item())
                    val_f1.append(f1_score(pred[i:i + 1], segs[i:i + 1]).item())
                    val_hd95.append(hd95_score(pred[i:i + 1], segs[i:i + 1]))

        val_metrics['dice'].append(np.nanmean(val_dice))
        val_metrics['iou'].append(np.nanmean(val_iou))
        val_metrics['precision'].append(np.nanmean(val_prec))
        val_metrics['recall'].append(np.nanmean(val_rec))
        val_metrics['f1'].append(np.nanmean(val_f1))
        val_metrics['hd95'].append(np.nanmean(val_hd95))

        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        print(f"\nrain metrics:")
        print(f"  Dice:      {train_metrics['dice'][-1]:.4f}")
        print(f"  IoU:       {train_metrics['iou'][-1]:.4f}")
        print(f"  Precision: {train_metrics['precision'][-1]:.4f}")
        print(f"  Recall:    {train_metrics['recall'][-1]:.4f}")
        print(f"  F1:        {train_metrics['f1'][-1]:.4f}")
        print(f"  HD95:      {train_metrics['hd95'][-1]:.4f}")
        print(f"\nValidation metrics:")
        print(f"  Dice:      {val_metrics['dice'][-1]:.4f}")
        print(f"  IoU:       {val_metrics['iou'][-1]:.4f}")
        print(f"  Precision: {val_metrics['precision'][-1]:.4f}")
        print(f"  Recall:    {val_metrics['recall'][-1]:.4f}")
        print(f"  F1:        {val_metrics['f1'][-1]:.4f}")
        print(f"  HD95:      {val_metrics['hd95'][-1]:.4f}")

        if val_metrics['dice'][-1] > best_val_dice:
            best_val_dice = val_metrics['dice'][-1]
            torch.save(model.state_dict(), best_model_path)
            print(f"\nBest model updated with Val Dice={best_val_dice:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs!")
                break

    torch.save(model.state_dict(), last_model_path)
    print(f"\nTraining finished!")
    print(f"Last model saved to {last_model_path}")
    print(f"Best model saved to {best_model_path}")

    plot_metrics(train_metrics, val_metrics, PLOTS_DIR)
    print(f"Plots saved to {PLOTS_DIR}")


def test_model(model_path, test_loader, device, save_dir):
    print("Testing...")

    model = StandardUNet3D(in_channels=4, out_channels=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    test_dice, test_iou, test_prec, test_rec, test_f1, test_hd95 = [], [], [], [], [], []

    with torch.no_grad():
        for imgs, segs in test_loader:
            imgs, segs = imgs.to(device), segs.to(device)
            logits = model(imgs)
            pred = torch.sigmoid(logits)

            for i in range(imgs.size(0)):
                test_dice.append(dice_score(pred[i:i + 1], segs[i:i + 1]).item())
                test_iou.append(iou_score(pred[i:i + 1], segs[i:i + 1]).item())
                test_prec.append(precision_score(pred[i:i + 1], segs[i:i + 1]).item())
                test_rec.append(recall_score(pred[i:i + 1], segs[i:i + 1]).item())
                test_f1.append(f1_score(pred[i:i + 1], segs[i:i + 1]).item())
                test_hd95.append(hd95_score(pred[i:i + 1], segs[i:i + 1]))

    metrics = {
        'Dice': np.nanmean(test_dice),
        'IoU': np.nanmean(test_iou),
        'Precision': np.nanmean(test_prec),
        'Recall': np.nanmean(test_rec),
        'F1': np.nanmean(test_f1),
        'HD95': np.nanmean(test_hd95)
    }

    std_metrics = {
        'Dice': np.nanstd(test_dice),
        'IoU': np.nanstd(test_iou),
        'Precision': np.nanstd(test_prec),
        'Recall': np.nanstd(test_rec),
        'F1': np.nanstd(test_f1),
        'HD95': np.nanstd(test_hd95)
    }

    print("\nTest Results:")
    for metric_name, mean_val in metrics.items():
        std_val = std_metrics[metric_name]
        print(f"{metric_name:12s}: {mean_val:.4f} ± {std_val:.4f}")

    os.makedirs(save_dir, exist_ok=True)
    results_file = os.path.join(save_dir, "test_results.txt")
    with open(results_file, 'w') as f:
        f.write("Test Results:\n")
        for metric_name, mean_val in metrics.items():
            std_val = std_metrics[metric_name]
            f.write(f"{metric_name:12s}: {mean_val:.4f} ± {std_val:.4f}\n")

    fig, ax = plt.subplots(figsize=(12, 6))

    metric_names = list(metrics.keys())
    metric_values = [metrics[m] for m in metric_names]
    metric_stds = [std_metrics[m] for m in metric_names]

    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12', '#95a5a6']

    bars = ax.bar(metric_names, metric_values, yerr=metric_stds,
                  capsize=5, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Test Set Performance - All Metrics', fontsize=14, fontweight='bold')
    ax.set_ylim([0, max(metric_values) * 1.2])
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    for bar, val, std in zip(bars, metric_values, metric_stds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + std,
                f'{val:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    chart_path = os.path.join(save_dir, 'test_metrics_summary.png')
    plt.savefig(chart_path, dpi=150)
    plt.close()

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    all_scores = [test_dice, test_iou, test_prec, test_rec, test_f1, test_hd95]

    for idx, (metric_name, scores) in enumerate(zip(metric_names, all_scores)):
        ax = axes[idx]
        bp = ax.boxplot([scores], labels=[metric_name], patch_artist=True,
                        boxprops=dict(facecolor=colors[idx], alpha=0.7),
                        medianprops=dict(color='red', linewidth=2),
                        whiskerprops=dict(linewidth=1.5),
                        capprops=dict(linewidth=1.5))

        ax.set_ylabel('Score', fontsize=10, fontweight='bold')
        ax.set_title(f'{metric_name} Distribution', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    boxplot_path = os.path.join(save_dir, 'test_metrics_distribution.png')
    plt.savefig(boxplot_path, dpi=150)
    plt.close()
    return metrics


if __name__ == '__main__':
    main_train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_ds = BratsDataset(TEST_DIR, target_shape=TARGET_SHAPE, normalization=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    best_model_path = os.path.join(model_dir, "best_model.pth")
    if os.path.exists(best_model_path):
        test_results = test_model(best_model_path, test_loader, device, PLOTS_DIR)
    else:
        print(f" Best model not found: {best_model_path}")