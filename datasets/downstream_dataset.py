import os, glob
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class TennisDownstreamDataset(Dataset):
    """
    Labeled FH/BH dataset for classifier training.
    """

    def __init__(self, root, split="train", transform=None, val_ratio=0.2):
        self.transform = transform

        fore = glob.glob(os.path.join(root, "forehand", "*"))
        back = glob.glob(os.path.join(root, "backhand", "*"))

        X = fore + back
        y = [0]*len(fore) + [1]*len(back)

        train_X, val_X, train_y, val_y = train_test_split(
            X, y, test_size=val_ratio, random_state=42, stratify=y)

        if split == "train":
            self.paths = train_X
            self.labels = train_y
        else:
            self.paths = val_X
            self.labels = val_y

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label
