import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.models as models

from datasets.downstream_dataset import TennisDownstreamDataset


def evaluate(encoder, head, loader, criterion, device):
    encoder.eval()
    head.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            feats = encoder(imgs)
            logits = head(feats)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

    return total_loss / len(loader), total_correct / total_samples


def train():
    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    train_ds = TennisDownstreamDataset("data_downstream", split="train", transform=transform)
    val_ds = TennisDownstreamDataset("data_downstream", split="val", transform=transform)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)

    # Encoder backbone must match JePA pretraining (ResNet-18)
    encoder = models.resnet18(weights=None)
    in_dim = encoder.fc.in_features
    encoder.fc = nn.Identity()

    # Pick which JePA checkpoint to load
    jepa_epoch = 10  # you can try 8, 9, 10, etc.
    ckpt_path = f"checkpoints_ssl/resnet18_epoch{jepa_epoch}.pth"
    print(f"\nLoading JePA checkpoint: {ckpt_path}\n")

    state_dict = torch.load(ckpt_path, map_location=device)
    encoder.load_state_dict(state_dict)
    encoder.to(device)

    # New classification head for FH/BH
    head = nn.Linear(in_dim, 2)
    head.to(device)

    criterion = nn.CrossEntropyLoss()

    # ---------- Linear probe ----------
    print("\nLinear Probe (JePA)\n")

    for p in encoder.parameters():
        p.requires_grad = False

    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)

    for epoch in range(5):
        head.train()
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            with torch.no_grad():
                feats = encoder(imgs)

            logits = head(feats)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        val_loss, val_acc = evaluate(encoder, head, val_loader, criterion, device)
        print(f"[JePA] Linear Epoch {epoch+1}/5  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

    # ---------- Fine-tuning ----------
    print("\nFine-Tuning (JePA, 20 epochs)\n")

    for p in encoder.parameters():
        p.requires_grad = True

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(head.parameters()),
        lr=5e-5
    )

    best_val_acc = 0.0
    best_state = None

    for epoch in range(20):
        encoder.train()
        head.train()

        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            feats = encoder(imgs)
            logits = head(feats)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        val_loss, val_acc = evaluate(encoder, head, val_loader, criterion, device)
        print(f"[JePA] FT Epoch {epoch+1}/20  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {
                "encoder": encoder.state_dict(),
                "head": head.state_dict()
            }

    print(f"\n[JePA] Best Fine-Tuning Accuracy: {best_val_acc:.4f}")

    if best_state is not None:
        torch.save(best_state, "checkpoints_ssl/best_ft_model_jepa.pth")
        print("Saved → checkpoints_ssl/best_ft_model_jepa.pth")


if __name__ == "__main__":
    train()
