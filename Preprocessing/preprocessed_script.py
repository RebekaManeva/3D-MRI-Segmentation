import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm


def z_score_normalize(volume):
    brain_mask = volume > 0
    if brain_mask.sum() == 0:
        return volume

    mean = volume[brain_mask].mean()
    std = volume[brain_mask].std()

    normalized = volume.copy()
    if std > 0:
        normalized[brain_mask] = (volume[brain_mask] - mean) / std

    return normalized


def compute_tight_bbox(volume, margin=5):
    coords = np.argwhere(volume > 0)

    if len(coords) == 0:
        return None

    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    mins = np.maximum(0, mins - margin)
    maxs = np.minimum(volume.shape, maxs + margin + 1)

    return tuple(zip(mins, maxs))


def preprocess_brats_simple(input_folder, output_folder, margin=5):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True, parents=True)

    patient_folders = [f for f in input_folder.iterdir() if f.is_dir()]

    print(f"Found {len(patient_folders)} patients\n")

    for folder in tqdm(patient_folders, desc="Processing"):
        folder_name = folder.name
        try:
            t1_path = folder / f"{folder_name}-t1n.nii.gz"
            t1c_path = folder / f"{folder_name}-t1c.nii.gz"
            t2_path = folder / f"{folder_name}-t2w.nii.gz"
            flair_path = folder / f"{folder_name}-t2f.nii.gz"
            mask_path = folder / f"{folder_name}-seg.nii.gz"

            if not all([t1_path.exists(), t1c_path.exists(),
                        t2_path.exists(), flair_path.exists(), mask_path.exists()]):
                print(f"Files missing {folder_name}")
                continue

            t1 = nib.load(t1_path).get_fdata()
            t1c = nib.load(t1c_path).get_fdata()
            t2 = nib.load(t2_path).get_fdata()
            flair = nib.load(flair_path).get_fdata()
            mask = nib.load(mask_path).get_fdata()
            bbox = compute_tight_bbox(t1, margin=margin)

            if bbox is None:
                continue

            (x0, x1), (y0, y1), (z0, z1) = bbox
            t1 = t1[x0:x1, y0:y1, z0:z1]
            t1c = t1c[x0:x1, y0:y1, z0:z1]
            t2 = t2[x0:x1, y0:y1, z0:z1]
            flair = flair[x0:x1, y0:y1, z0:z1]
            mask = mask[x0:x1, y0:y1, z0:z1]

            t1 = z_score_normalize(t1)
            t1c = z_score_normalize(t1c)
            t2 = z_score_normalize(t2)
            flair = z_score_normalize(flair)
            multi_channel = np.stack([t1, t1c, t2, flair], axis=-1)
            patient_output = output_folder / folder_name
            patient_output.mkdir(exist_ok=True, parents=True)

            np.save(patient_output / "data.npy", multi_channel.astype(np.float32))
            np.save(patient_output / "mask.npy", mask.astype(np.uint8))

        except Exception as e:
            print(f"Error in {folder_name}: {e}")
            continue

if __name__ == "__main__":
    input_folder = "#"
    output_folder = "#"

    preprocess_brats_simple(
        input_folder=input_folder,
        output_folder=output_folder,
        margin=5)


