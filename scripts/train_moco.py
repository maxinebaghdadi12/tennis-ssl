import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.lejepa_dataset import LeJEPADataset
from transforms.moco_transforms import TwoCropsTransform
from models.moco_model import MoCoV2


def main():
    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    transform = TwoCropsTransform()
    dataset = LeJEPADataset("data_ssl", transform=transform)

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    model = MoCoV2(dim=128, K=8192, m=0.999, T=0.2)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.encoder_q.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )

    os.makedirs("checkpoints_moco", exist_ok=True)

    epochs = 20
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for views in loader:
            im_q = views[0].to(device)
            im_k = views[1].to(device)

            logits, labels = model(im_q, im_k)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(loader)
        print(f"Epoch {epoch+1}/{epochs} - MoCo loss: {avg_loss:.4f}")

        ckpt = f"checkpoints_moco/moco_resnet18_epoch{epoch+1}.pth"
        torch.save(model.encoder_q.state_dict(), ckpt)
        print(f"Saved checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
