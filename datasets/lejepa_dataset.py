import os
import glob
from typing import List, Callable, Optional

from PIL import Image
from torch.utils.data import Dataset


class LeJEPADataset(Dataset):
    """
    Unlabeled dataset used for SSL pretraining (JePA or MoCo).

    Expects images under:
        root/images/*
    """

    def __init__(self, root: str, transform: Optional[Callable] = None):
        self.root = root
        self.transform = transform

        self.paths: List[str] = glob.glob(os.path.join(root, "images", "*"))
        if len(self.paths) == 0:
            raise RuntimeError(f"No images found in {root}/images/")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")

        if self.transform:
            return self.transform(img)

        return img
