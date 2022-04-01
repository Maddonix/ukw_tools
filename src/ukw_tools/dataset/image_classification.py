from torch.utils.data import Dataset
import cv2
import torch
from .utils import img_augmentations, crop_img, img_transforms
import numpy as np
import warnings


class ImageClassificationDs(Dataset):
    def __init__(self, paths, labels, crop, scaling: int = 75, training: bool = True, skip_augmentation = False, **kwargs):
        self.paths = paths
        self.scaling = scaling
        self.crop = crop
        self.training = training
        self.skip_augmentation = skip_augmentation
        if "swapaxes" in kwargs:
            self.swapaxes = kwargs["swapaxes"]
        else: self.swapaxes = None

        self.labels = labels
        # assert len(paths) == len(labels)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = cv2.imread(self.paths[idx])
            img = crop_img(img, self.crop[idx])
        except:
            warnings.warn(f"Failed to read image {self.paths[idx]}")
            img = np.zeros([1024,1024,3], np.uint8)
        
        width = int(1024 * self.scaling / 100)
        height = int(1024 * self.scaling / 100)
        dim = (width, height)
        
        # FIXME INTER AREA?
        img = cv2.resize(img, dsize=dim, interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.training:
            if not self.skip_augmentation:
                img = img_augmentations(image=img)["image"]

        img = img_transforms(image=img)["image"]
        img = torch.tensor(img)
        img = torch.swapaxes(img, 0, 2)

        return img, self.labels[idx]