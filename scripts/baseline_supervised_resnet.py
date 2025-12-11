import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms as T

from sklearn.metrics import f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from datasets.downstream_dataset import TennisDownstreamDataset


def evaluate(model, loader, criterion, device):
    model.eval()
    
    all_preds, all_labels = [], []
    total_loss = 0.0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            logits = model(imgs)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    f1 = f1_score(all_labels, all_preds, average="binary")

    return avg_loss, acc, f1, all_preds, all_labels


def main():

    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
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
    val_ds   = TennisDownstreamDataset("datasets/data_downstream", split="val",   transform=transform)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=16, shuffle=False)

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)  
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_acc = 0
    best_state = None

    print("\nTraining Supervised ResNet Baseline\n")
    for epoch in range(15):
        model.train()
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        val_loss, val_acc, val_f1, preds, labels = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1:02d}/15  loss={val_loss:.4f}  acc={val_acc:.4f}  f1={val_f1:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {
                "model": model.state_dict(),
                "preds": preds,
                "labels": labels
            }

    print(f"\nBest Supervised Accuracy: {best_acc:.4f}")

    os.makedirs("checkpoints/supervised_baseline", exist_ok=True)
    save_path = "checkpoints/supervised_baseline/resnet_baseline.pth"
    torch.save(best_state, save_path)
    print(f"Saved → {save_path}")

    cm = confusion_matrix(best_state["labels"], best_state["preds"])
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Forehand", "Backhand"],
        yticklabels=["Forehand", "Backhand"]
    )
    plt.title("Supervised ResNet Baseline – Confusion Matrix")
    plt.savefig("figures/supervised_confusion_matrix.png")
    plt.close()

    print("Saved confusion matrix → figures/supervised_confusion_matrix.png")


if __name__ == "__main__":
    main()
