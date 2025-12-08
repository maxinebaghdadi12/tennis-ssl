import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.models as models

from datasets.downstream_dataset import TennisDownstreamDataset


def evaluate(encoder, head, loader, criterion, device):
    """Evaluate encoder + classifier head."""
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

    # Load encoder with MoCo projection head
    encoder = models.resnet18(weights=None)
    in_dim = encoder.fc.in_features
    encoder.fc = nn.Sequential(
        nn.Linear(in_dim, in_dim),
        nn.ReLU(inplace=True),
        nn.Linear(in_dim, 128)
    )

    moco_epoch = 20
    ckpt_path = f"checkpoints_moco/moco_resnet18_epoch{moco_epoch}.pth"
    print(f"\nLoading MoCo checkpoint: {ckpt_path}\n")

    state_dict = torch.load(ckpt_path, map_location=device)
    encoder.load_state_dict(state_dict)

    # Remove MoCo head for downstream classification
    encoder.fc = nn.Identity()
    encoder.to(device)

    head = nn.Linear(in_dim, 2)
    head.to(device)

    criterion = nn.CrossEntropyLoss()

    # Linear probe (encoder frozen)
    print("\nLinear Probe\n")

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
        print(f"Linear Epoch {epoch+1}/5  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

    # Fine-tuning
    print("\nFine-Tuning (20 epochs)\n")

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
        print(f"FT Epoch {epoch+1}/20  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {
                "encoder": encoder.state_dict(),
                "head": head.state_dict()
            }

    print(f"\nBest Fine-Tuning Accuracy: {best_val_acc:.4f}")

    if best_state is not None:
        torch.save(best_state, "checkpoints_moco/best_ft_model.pth")
        print("Saved → checkpoints_moco/best_ft_model.pth")


if __name__ == "__main__":
    train()