import os
import warnings
import numpy as np
import torch
import matplotlib.pyplot as plt
import nibabel as nib

from monai.networks.nets import SegResNet

warnings.filterwarnings("ignore")

DEVICE = torch.device("cpu")
BASE = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = r"brats\data_normalized"

MODEL_PATH = os.path.join(BASE, "..", "MOTUM-v.2.2", "best_segresnet.pth")

OUT_DIR = os.path.join(BASE, "plots_brats_eval")
os.makedirs(OUT_DIR, exist_ok=True)

PATCH_SIZE = (96, 96, 96)  # како MOTUM patch


def match_histogram_to_motum(image: np.ndarray) -> np.ndarray:
    """
    Приближно нормализирање како MOTUM:
    од [-3, 3] до [0, 1].
    """
    image = np.clip(image, -3, 3)
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    return image


def center_crop_3d(vol: np.ndarray, size=(96, 96, 96)) -> np.ndarray:
    d, h, w = vol.shape
    cd, ch, cw = size

    sd = max((d - cd) // 2, 0)
    sh = max((h - ch) // 2, 0)
    sw = max((w - cw) // 2, 0)

    ed = sd + cd
    eh = sh + ch
    ew = sw + cw

    return vol[sd:ed, sh:eh, sw:ew]


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

    img = im_nii.get_fdata().astype(np.float32)
    lab = lab_nii.get_fdata().astype(np.int16)

    img = match_histogram_to_motum(img)  # !!!

    img_crop = center_crop_3d(img, PATCH_SIZE)
    lab_crop = center_crop_3d(lab, PATCH_SIZE)

    img_t = torch.from_numpy(img_crop[None, None, ...]).float().to(DEVICE)

    with torch.no_grad():
        logits = model(img_t)
        probs = torch.softmax(logits, dim=1)
        pred_cls = probs.argmax(1).cpu().numpy()[0]

    pred_fg = (pred_cls > 0).astype(np.uint8)
    lab_fg = (lab_crop > 0).astype(np.uint8)

    dice_fg = dice_coefficient(pred_fg, lab_fg)

    if idx == 0:
        print("  Dice (foreground, cropped):",
              f"{dice_fg:.4f}" if not np.isnan(dice_fg) else "NaN")
        print("-" * 50)

    if not np.isnan(dice_fg):
        all_dice.append(dice_fg)
        case_ids.append(pid)

    print(f"[{idx + 1}/{len(pairs)}] {pid}: Dice_fg = "
          f"{dice_fg:.4f}" if not np.isnan(dice_fg) else
          f"[{idx + 1}/{len(pairs)}] {pid}: Dice_fg = NaN (no foreground)")

if len(all_dice) == 0:
    print("\nNo valid Dice scores (all cases had no foreground in both GT and prediction).")
else:
    all_dice = np.array(all_dice, dtype=np.float32)

    plt.figure(figsize=(12, 4))
    x = np.arange(len(all_dice))
    plt.bar(x, all_dice)
    plt.xlabel("Case index (valid foreground cases)")
    plt.ylabel("Foreground Dice (center 96³)")
    plt.ylim(0.0, 1.0)
    plt.title(f"Dice per case on BraTS (n={len(all_dice)})")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    dice_plot_path = os.path.join(OUT_DIR, "dice_per_case_bar.png")
    plt.savefig(dice_plot_path)
    plt.close()

    print()
    print("----------------------------------------------------------")
    print(f"Valid cases with foreground Dice: {len(all_dice)}")  # 74
    print(f"Mean Dice (foreground, cropped): {np.mean(all_dice):.4f}")  # 0.3009
    print(f"Std Dice : {np.std(all_dice):.4f}")  # 0.2774
    print(f"Min Dice : {np.min(all_dice):.4f}")  # 0.0000
    print(f"Max Dice : {np.max(all_dice):.4f}")  # 0.8334
