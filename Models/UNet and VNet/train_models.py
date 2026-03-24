import os
import random
import torch.optim as optim
from metrics import *
from UNet_model import UNet3D
from VNet_model import VNet3D
from load_dataset import BratsDataset
from torch.utils.data import DataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(BASE_DIR, "data")
TRAIN_DIR = os.path.join(DATA_ROOT, "Train")
VAL_DIR   = os.path.join(DATA_ROOT, "Validation")
TEST_DIR  = os.path.join(DATA_ROOT, "Test")
PLOTS_DIR  = os.path.join(BASE_DIR, "plots")
MODELS_DIR = os.path.join(BASE_DIR, "models")

TARGET_SHAPE = (128, 128, 128)
BATCH_SIZE = 2
NUM_WORKERS = 2
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100

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

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("\nData loading...")
    train_ds = BratsDataset(TRAIN_DIR, target_shape=TARGET_SHAPE, normalization=True)
    val_ds = BratsDataset(VAL_DIR, target_shape=TARGET_SHAPE, normalization=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples: {len(val_ds)}")

    model = UNet3D(in_channels=4, out_channels=3).to(device)
    #model = VNet3D(in_channels=4, out_channels=3).to(device)
    criterion = DiceBCELoss(dice_weight=0.8, bce_weight=0.2)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    best_model_path = os.path.join(MODELS_DIR, "best_model.pth")
    last_model_path = os.path.join(MODELS_DIR, "last_model.pth")

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
        print(f"\nTrain metrics:")
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


def test_model(model_path, test_loader, device, save_dir):
    print("Testing...")

    model = UNet3D(in_channels=4, out_channels=3).to(device)
    #model = VNet3D(in_channels=4, out_channels=3).to(device)
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
    return metrics


if __name__ == '__main__':
    main_train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_ds = BratsDataset(TEST_DIR, target_shape=TARGET_SHAPE, normalization=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    best_model_path = os.path.join(MODELS_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        test_results = test_model(best_model_path, test_loader, device, PLOTS_DIR)
    else:
        print(f" Best model not found: {best_model_path}")