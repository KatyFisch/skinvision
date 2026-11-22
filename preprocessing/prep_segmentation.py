import numpy as np
import torch

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class ISICDataset(Dataset):
    def __init__(self, df, image_size=(512, 384)):

        self.df = df.reset_index(drop=True)
        self.image_size = image_size

        # ImageNet normalization
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        # Load image and mask
        image = Image.open("../data" / row["path"]).convert("RGB")
        mask = Image.open("../data" / row["path_segmentation"]).convert("L")

        # Resize
        image = image.resize(self.image_size, Image.Resampling.BILINEAR)

        mask = mask.resize(self.image_size, Image.Resampling.NEAREST)

        # Convert image to numpy
        image = np.array(image).astype(np.float32) / 255.0

        # Normalize image
        image = ((image - self.mean) / self.std).astype(np.float32)

        # Convert mask to binary
        mask = np.array(mask).astype(np.float32)
        mask = (mask > 127).astype(np.float32)

        # Convert to PyTorch tensors
        image = torch.from_numpy(image).float().permute(2, 0, 1)
        mask = torch.from_numpy(mask).float().unsqueeze(0)

        return image, mask


def create_dataloaders(df_train, df_val, df_test, batch_size=8, image_size=(512, 384), num_workers=0):

    train_dataset = ISICDataset(df_train, image_size)
    val_dataset = ISICDataset(df_val, image_size)
    test_dataset = ISICDataset(df_test, image_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
