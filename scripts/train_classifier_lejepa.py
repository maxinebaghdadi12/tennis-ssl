import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.models as models

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, confusion_matrix

from datasets.downstream_dataset import TennisDownstreamDataset



def evaluate(encoder, head, loader, criterion, device):
    encoder.eval()
    head.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            feats = encoder(imgs)
            logits = head(feats)
            loss = criterion(logits, labels)

            total_loss += loss.item()

            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    f1 = f1_score(all_labels, all_preds, average="binary")

    return avg_loss, acc, f1, all_preds, all_labels


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

    train_ds = TennisDownstreamDataset("datasets/data_downstream", split="train", transform=transform)
    val_ds = TennisDownstreamDataset("datasets/data_downstream", split="val", transform=transform)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    encoder = models.resnet18(weights=None)
    in_dim = encoder.fc.in_features
    encoder.fc = nn.Identity()

    ckpt_path = "checkpoints/checkpoints_ssl/resnet18_epoch10.pth"
    print(f"\nLoading JePA checkpoint: {ckpt_path}\n")

    state_dict = torch.load(ckpt_path, map_location=device)
    encoder.load_state_dict(state_dict)
    encoder.to(device)

    head = nn.Linear(in_dim, 2)
    head.to(device)

    criterion = nn.CrossEntropyLoss()

    print("\n[JePA] Linear Probe\n")

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

        val_loss, val_acc, val_f1, _, _ = evaluate(encoder, head, val_loader, criterion, device)
        print(f"[JePA] Linear Epoch {epoch+1}/5  loss={val_loss:.4f}  acc={val_acc:.4f}  f1={val_f1:.4f}")

    print("\n[JePA] Fine-Tuning (20 epochs)\n")

    for p in encoder.parameters():
        p.requires_grad = True

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(head.parameters()),
        lr=5e-5
    )

    best_acc = 0
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

        val_loss, val_acc, val_f1, preds, labels = evaluate(encoder, head, val_loader, criterion, device)
        print(f"[JePA] FT Epoch {epoch+1}/20  loss={val_loss:.4f}  acc={val_acc:.4f}  f1={val_f1:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {
                "encoder": encoder.state_dict(),
                "head": head.state_dict(),
                "preds": preds,
                "labels": labels
            }

    print(f"\n[JePA] Best Fine-Tuning Accuracy: {best_acc:.4f}")

    save_path = "checkpoints/checkpoints_ssl/best_ft_model_jepa.pth"
    torch.save(best_state, save_path)
    print(f"Saved → {save_path}")

    # Confusion matrix
    cm = confusion_matrix(best_state["labels"], best_state["preds"])
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Forehand", "Backhand"],
                yticklabels=["Forehand", "Backhand"])
    plt.title("JePA Confusion Matrix")
    plt.savefig("jepa_confusion_matrix.png")
    plt.close()


if __name__ == "__main__":
    train()
