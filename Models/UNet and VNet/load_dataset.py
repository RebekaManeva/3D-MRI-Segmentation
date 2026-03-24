import numpy as np
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import zoom
import os
import torch

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
            target_shape[2] / current_shape[2]]

        if volume.ndim == 4:
            zoom_factors.append(1)
        resized = zoom(volume, zoom_factors, order=1)
        return resized

    def resize_mask(self, mask, target_shape):
        current_shape = mask.shape
        zoom_factors = [
            target_shape[0] / current_shape[0],
            target_shape[1] / current_shape[1],
            target_shape[2] / current_shape[2]]

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