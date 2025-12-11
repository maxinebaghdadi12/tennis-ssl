import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models

import lejepa
from datasets.lejepa_dataset import LeJEPADataset
from scripts.lejepa_transforms import LeJEPAMultiCrop


def main():
    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    transform = LeJEPAMultiCrop()
    dataset = LeJEPADataset("datasets/data/data_ssl", transform=transform)

    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    encoder = models.resnet18(weights=None)
    encoder.fc = nn.Identity()
    encoder.to(device)

    univariate_test = lejepa.univariate.EppsPulley()
    sigreg = lejepa.multivariate.SlicingUnivariateTest(
        univariate_test=univariate_test,
        num_slices=256,
    )

    optimizer = torch.optim.Adam(
        encoder.parameters(),
        lr=1e-3,
        weight_decay=5e-4,
    )

    os.makedirs("checkpoints_ssl", exist_ok=True)
    epochs = 10
    encoder.train()

    for epoch in range(epochs):
        epoch_loss = 0.0

        for views in loader:
            embeddings = []
            for crop in views:
                crop = crop.to(device)
                feats = encoder(crop)
                embeddings.append(feats)

            embeddings = torch.cat(embeddings, dim=0)
            loss = sigreg(embeddings)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

        ckpt_path = f"checkpoints_ssl/resnet18_epoch{epoch+1}.pth"
        torch.save(encoder.state_dict(), ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
