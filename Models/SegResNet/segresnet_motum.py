import os, json, time, warnings
import torch
import torch.nn.functional as F
import numpy as np
from monai.data import CacheDataset, DataLoader
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    NormalizeIntensityd, RandFlipd, RandAffined,
    RandGaussianNoised, RandAdjustContrastd, RandScaleIntensityd,
    RandGaussianSmoothd, EnsureTyped,
    RandCropByPosNegLabeld, SpatialPadd, AsDiscrete
)
from monai.losses import DiceLoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.inferers import sliding_window_inference
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score
from torch.amp import autocast, GradScaler
from monai.networks.nets import SegResNet
from visualize_results import plot_metrics

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))
DATASET_JSON = f"{BASE}/data/dataset.json"
SPLITS_JSON = f"{BASE}/data/splits/split_single.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(DATASET_JSON) as f:
    DS = json.load(f)
with open(SPLITS_JSON) as f:
    SPLITS = [json.load(f)]


def build_lists(fold: int):
    all_items = DS["training"]
    fold_info = SPLITS[fold]
    tr_ids, va_ids = set(fold_info["train"]), set(fold_info["val"])
    train_list, val_list = [], []
    for it in all_items:
        sample = {
            "pid": it["patient_id"],
            "im": it["images"]["t1"],
            "label_ce": it["labels"]["ce_core"],
            "label_fl": it["labels"]["flair_abn"],
        }
        (train_list if it["patient_id"] in tr_ids else val_list).append(sample)
    return train_list, val_list


def get_transforms(patch_size=(128, 128, 128)):
    common = [
        LoadImaged(keys=["im", "label_ce", "label_fl"]),
        EnsureChannelFirstd(keys=["im", "label_ce", "label_fl"]),
        Orientationd(keys=["im", "label_ce", "label_fl"], axcodes="RAS"),
        Spacingd(keys=["im", "label_ce", "label_fl"], pixdim=(1, 1, 1), mode=("bilinear", "nearest", "nearest")),
        NormalizeIntensityd(keys=["im"], nonzero=True, channel_wise=True),
        EnsureTyped(keys=["im", "label_ce", "label_fl"]),
    ]
    train_aug = [
        SpatialPadd(keys=["im", "label_ce", "label_fl"], spatial_size=patch_size),
        RandCropByPosNegLabeld(keys=["im", "label_ce", "label_fl"], label_key="label_ce",
                               spatial_size=patch_size, pos=4, neg=1, num_samples=2),
        RandFlipd(keys=["im", "label_ce", "label_fl"], prob=0.5, spatial_axis=(0, 1, 2)),
        RandAffined(keys=["im", "label_ce", "label_fl"], prob=0.3, rotate_range=(0.1, 0.1, 0.1),
                    scale_range=(0.1, 0.1, 0.1), mode=("bilinear", "nearest", "nearest")),
        RandAdjustContrastd(keys=["im"], prob=0.15, gamma=(0.7, 1.5)),
        RandGaussianNoised(keys=["im"], prob=0.15, mean=0.0, std=0.01),
        RandScaleIntensityd(keys=["im"], factors=0.1, prob=0.2),
        RandGaussianSmoothd(keys=["im"], sigma_x=(0.5, 1.5), prob=0.15),
        EnsureTyped(keys=["im", "label_ce", "label_fl"]),
    ]
    val_aug = [
        SpatialPadd(keys=["im", "label_ce", "label_fl"], spatial_size=patch_size),
        RandCropByPosNegLabeld(keys=["im", "label_ce", "label_fl"], label_key="label_ce",
                               spatial_size=patch_size, pos=4, neg=1, num_samples=1),
        EnsureTyped(keys=["im", "label_ce", "label_fl"]),
    ]
    return Compose(common + train_aug), Compose(common + val_aug)


def merge_labels_collate(batch):
    from torch.utils.data.dataloader import default_collate
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    b = default_collate(batch)
    if isinstance(b, list):
        b = b[0]
    if not isinstance(b, dict):
        return None
    label = torch.zeros_like(b["label_ce"], dtype=torch.long).squeeze(1)
    fl = b["label_fl"].squeeze(1).long()
    ce = b["label_ce"].squeeze(1).long()
    label = label + (fl > 0).long() * 1
    label = torch.where(ce > 0, torch.tensor(2, device=label.device), label)
    b["label"] = label
    b.pop("label_ce", None)
    b.pop("label_fl", None)
    return b


def make_dataloaders(fold=0, patch_size=(128, 128, 128), batch_size=1, num_workers=2, cache_rate=0.0):
    tr, va = build_lists(fold)
    tr_tf, va_tf = get_transforms(patch_size)
    tr_ds = CacheDataset(tr, tr_tf, cache_rate, num_workers)
    va_ds = CacheDataset(va, va_tf, cache_rate, num_workers)
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                           pin_memory=True, collate_fn=merge_labels_collate)
    va_loader = DataLoader(va_ds, batch_size=1, shuffle=False, num_workers=num_workers,
                           pin_memory=True, collate_fn=merge_labels_collate)
    return tr_loader, va_loader


def make_model():
    model = SegResNet(spatial_dims=3, init_filters=32, in_channels=1, out_channels=3,
                      dropout_prob=0.1, norm="instance")
    return model.to(DEVICE)


def make_loss():
    dice = DiceLoss(to_onehot_y=True, softmax=True, include_background=True)
    weights = torch.tensor([0.05, 0.35, 0.6]).to(DEVICE)

    def combined_loss(pred, target):
        loss_dice = dice(pred, target)
        loss_ce = F.cross_entropy(pred, target.squeeze(1).long(), weight=weights)
        return 0.5 * loss_dice + 0.5 * loss_ce

    return combined_loss


def make_optim(model, lr=1e-4, wd=1e-5):
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)


scaler = GradScaler("cuda")


def train_one_epoch(model, loader, opt, loss_fn):
    model.train();
    run = 0.0
    for b in loader:
        imgs = b["im"].to(DEVICE);
        lbl = b["label"].to(DEVICE).unsqueeze(1)
        opt.zero_grad()
        with autocast("cuda"):
            out = model(imgs);
            loss = loss_fn(out, lbl)
        scaler.scale(loss).backward();
        scaler.step(opt);
        scaler.update()
        run += loss.item()
    return run / max(1, len(loader))


@torch.no_grad()
def validate(model, loader, roi_size=(96, 96, 96)):
    model.eval()
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    total_time = 0.0
    dice_scores = []

    for batch in loader:
        if batch is None:
            continue
        imgs = batch["im"].to(DEVICE)
        labels = batch["label"].to(DEVICE).long()

        labels_onehot = torch.nn.functional.one_hot(labels, num_classes=3)
        labels_onehot = labels_onehot.permute(0, 4, 1, 2, 3).float()

        t0 = time.time()
        preds = sliding_window_inference(imgs, roi_size=roi_size, sw_batch_size=1, predictor=model)
        preds = torch.softmax(preds, dim=1)
        total_time += time.time() - t0

        dice_batch = dice_metric(y_pred=preds, y=labels_onehot)
        scores = dice_metric.aggregate()
        dice_metric.reset()

        valid_scores = scores[~torch.isnan(scores)]
        if len(valid_scores) > 0:
            dice_scores.append(valid_scores.mean().item())

    mean_dice = sum(dice_scores) / max(1, len(dice_scores))
    mean_time = total_time / max(1, len(loader))
    return mean_dice, mean_time


@torch.no_grad()
def evaluate_metrics(model, loader, patch_size=(96, 96, 96)):
    model.eval()
    dice_metric = DiceMetric(include_background=False, reduction="none")
    hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95)
    threshold = AsDiscrete(argmax=True)

    all_dice, all_hd95, all_f1, all_iou, all_precision, all_recall = [], [], [], [], [], []

    for batch in loader:
        if batch is None:
            continue
        imgs = batch["im"].to(DEVICE)
        labels = batch["label"].to(DEVICE).long()
        if torch.sum(labels) == 0:
            continue

        preds = sliding_window_inference(imgs, roi_size=patch_size, sw_batch_size=1, predictor=model)
        preds = torch.softmax(preds, dim=1)
        preds_disc = threshold(preds)

        labels_onehot = torch.nn.functional.one_hot(labels, num_classes=3)
        labels_onehot = labels_onehot.permute(0, 4, 1, 2, 3).float()

        dsc = dice_metric(y_pred=preds_disc, y=labels_onehot)
        hd = hd95_metric(y_pred=preds_disc, y=labels_onehot)

        mean_dsc = torch.nan_to_num(torch.nanmean(dsc), nan=0.0).item()
        mean_hd = torch.nan_to_num(torch.nanmean(hd), nan=0.0).item()
        all_dice.append(mean_dsc)
        all_hd95.append(mean_hd)

        y_true = labels.cpu().numpy().ravel()
        y_pred = preds_disc.argmax(1).cpu().numpy().ravel()

        all_precision.append(precision_score(y_true, y_pred, average='macro', zero_division=0))
        all_recall.append(recall_score(y_true, y_pred, average='macro', zero_division=0))
        all_f1.append(f1_score(y_true, y_pred, average='macro', zero_division=0))
        all_iou.append(jaccard_score(y_true, y_pred, average='macro', zero_division=0))

    metrics = {
        "Dice": np.mean(all_dice),
        "F1": np.mean(all_f1),
        "IoU": np.mean(all_iou),
        "Precision": np.mean(all_precision),
        "Recall": np.mean(all_recall),
        "HD95": np.mean(all_hd95)
    }
    metrics = {k: np.nan_to_num(v, nan=0.0) for k, v in metrics.items()}
    return metrics


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()

    EPOCHS = 100
    PATCH = (96, 96, 96)
    BATCH = 1
    BASE_FILTERS = 32
    LR = 1e-4

    tr, va = make_dataloaders(fold=0, patch_size=PATCH, batch_size=BATCH, num_workers=2, cache_rate=0.0)
    model = make_model()
    loss_fn = make_loss()
    opt = make_optim(model, lr=LR)

    train_losses, val_dices = [], []
    epoch_metrics = {k: [] for k in ["Dice", "F1", "IoU", "Precision", "Recall", "HD95"]}
    best_dice = 0.0

    print(f"T1 | f={BASE_FILTERS} | lr={LR} | epochs={EPOCHS}")
    print("----------------------------------------------------------")
    for ep in range(1, EPOCHS + 1):
        tr_loss = train_one_epoch(model, tr, opt, loss_fn)
        val_dice, _ = validate(model, va, roi_size=PATCH)
        metrics = evaluate_metrics(model, va, patch_size=PATCH)
        train_losses.append(tr_loss)
        val_dices.append(val_dice)
        epoch_metrics["Dice"].append(val_dice)
        epoch_metrics["F1"].append(metrics["F1"])
        epoch_metrics["IoU"].append(metrics["IoU"])
        epoch_metrics["Precision"].append(metrics["Precision"])
        epoch_metrics["Recall"].append(metrics["Recall"])
        epoch_metrics["HD95"].append(metrics["HD95"])
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), os.path.join(BASE, "best_segresnet.pth"))
        print(f"Epoch {ep:03d}/{EPOCHS} | Train={tr_loss:.4f} | ValDice={val_dice:.4f} | "
              f"F1={metrics['F1']:.4f} | IoU={metrics['IoU']:.4f} | "
              f"Prec={metrics['Precision']:.4f} | Rec={metrics['Recall']:.4f} | "
              f"HD95={metrics['HD95']:.4f}")
    print("----------------------------------------------------------")
    print(f"Best Validation Dice: {best_dice:.4f}")
    plot_metrics(epoch_metrics, EPOCHS)
