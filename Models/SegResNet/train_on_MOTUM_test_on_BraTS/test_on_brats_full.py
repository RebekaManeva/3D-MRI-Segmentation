import os
import warnings
import numpy as np
import torch
import matplotlib.pyplot as plt
import nibabel as nib

from monai.networks.nets import SegResNet
from monai.inferers import sliding_window_inference

warnings.filterwarnings("ignore")

DEVICE = torch.device("cpu")

BASE = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = r"C:\Users\rebek\MANU\presmetki\brats\data_normalized"

MODEL_PATH = os.path.join(BASE, "..", "MOTUM-v.2.2", "best_segresnet.pth")

OUT_DIR = os.path.join(BASE, "plots_brats_full")
os.makedirs(OUT_DIR, exist_ok=True)

PATCH_SIZE = (96, 96, 96)
MAX_VISUAL = 6


def match_histogram_to_motum(image: np.ndarray) -> np.ndarray:
    """
    Приближно нормализирање како MOTUM:
    од [-3, 3] до [0, 1].
    """
    image = np.clip(image, -3, 3)
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    return image


def get_brats_pairs(data_dir: str):
    pairs = []
    for root, _, files in os.walk(data_dir):
        nii_files = [f for f in files if f.endswith(".nii.gz")]
        for f in nii_files:
            if "t1n" in f:
                base_id = f.split("-t1n")[0]
                seg_candidates = [s for s in nii_files if s.startswith(base_id) and "seg" in s]
                if len(seg_candidates) > 0:
                    seg_file = seg_candidates[0]
                    pairs.append({
                        "pid": base_id,
                        "im": os.path.join(root, f),
                        "label": os.path.join(root, seg_file)
                    })
    return pairs


def dice_coefficient(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    size1 = pred.sum()
    size2 = gt.sum()
    if size1 + size2 == 0:
        return np.nan
    return 2.0 * inter / (size1 + size2)


pairs = get_brats_pairs(DATA_PATH)
print(f"Total test pairs found: {len(pairs)}")
if len(pairs) == 0:
    raise SystemExit("No BraTS pairs found.")

model = SegResNet(
    spatial_dims=3,
    init_filters=32,
    in_channels=1,
    out_channels=3,
    dropout_prob=0.1,
    norm="instance"
)
state = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state)
model.to(DEVICE)
model.eval()

all_dice = []
case_ids = []

for idx, sample in enumerate(pairs):
    pid = sample["pid"]
    im_path = sample["im"]
    lab_path = sample["label"]

    im_nii = nib.load(im_path)
    lab_nii = nib.load(lab_path)

    img = im_nii.get_fdata().astype(np.float32)  # (H, W, D)
    lab = lab_nii.get_fdata().astype(np.int16)  # (H, W, D)

    # normalizacija
    img_norm = match_histogram_to_motum(img)

    img_t = torch.from_numpy(img_norm[None, None, ...]).float().to(DEVICE)

    with torch.no_grad():
        logits = sliding_window_inference(
            img_t,
            roi_size=PATCH_SIZE,
            sw_batch_size=1,
            predictor=model,
        )

        probs = torch.softmax(logits, dim=1)
        pred_cls = probs.argmax(1).cpu().numpy()[0]

    pred_fg = (pred_cls > 0).astype(np.uint8)
    lab_fg = (lab > 0).astype(np.uint8)

    dice_fg = dice_coefficient(pred_fg, lab_fg)
    case_ids.append(pid)

    if idx == 0:
        print(f"  Dice (foreground, full volume): {dice_fg:.4f}" if not np.isnan(dice_fg) else "  Dice: NaN")

    if not np.isnan(dice_fg):
        all_dice.append(dice_fg)
    else:
        all_dice.append(0.0)
    print(f"[{idx + 1}/{len(pairs)}] {pid}: Dice_fg (full) = {dice_fg:.4f}" if not np.isnan(dice_fg) else
          f"[{idx + 1}/{len(pairs)}] {pid}: Dice_fg = NaN (no foreground)")

    if idx < MAX_VISUAL:
        sums_per_slice = lab_fg.sum(axis=(0, 1))
        if sums_per_slice.sum() > 0:
            slice_idx = np.argmax(sums_per_slice)
        else:
            slice_idx = img_norm.shape[2] // 2

        t1_slice = img_norm[:, :, slice_idx]
        gt_slice = lab[:, :, slice_idx]
        pred_slice = pred_cls[:, :, slice_idx]

        plt.figure(figsize=(12, 4))
        plt.suptitle(f"Case {idx + 1} - {pid} - Dice_fg={dice_fg:.3f}", fontsize=12)

        plt.subplot(1, 3, 1)
        plt.imshow(t1_slice, cmap="gray")
        plt.title("T1n (full slice)")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(gt_slice, cmap="Reds")
        plt.title("GT (full slice)")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(pred_slice, cmap="Blues")
        plt.title("Pred (full slice)")
        plt.axis("off")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        fig_path = os.path.join(OUT_DIR, f"case_{idx + 1:03d}_{pid}_fullslice.png")
        plt.savefig(fig_path, dpi=150)
        plt.close()

all_dice = np.array(all_dice, dtype=np.float32)

if len(all_dice) == 0:
    print("\nNo valid Dice scores (all cases had no foreground in both GT and prediction).")
else:
    plt.figure(figsize=(10, 4))
    x = np.arange(len(all_dice))
    plt.bar(x, all_dice)
    plt.xlabel("Case index")
    plt.ylabel("Foreground Dice (full volume)")
    plt.ylim(0.0, 1.0)
    plt.title(f"Dice per case on BraTS (full volume, n={len(all_dice)})")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    dice_plot_path = os.path.join(OUT_DIR, "dice_per_case_bar.png")
    plt.savefig(dice_plot_path, dpi=150)
    plt.close()

    print("\n----------------------------------------------------------")
    print(f"Cases evaluated          : {len(all_dice)}")
    print(f"Mean Dice (full volume)  : {np.mean(all_dice):.4f}")  # 0.1669
    print(f"Std Dice                 : {np.std(all_dice):.4f}")  # 0.1786
    print(f"Min Dice                 : {np.min(all_dice):.4f}")  # 0.0000
    print(f"Max Dice                 : {np.max(all_dice):.4f}")  # 0.6733
