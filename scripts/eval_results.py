import os
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms as T

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from datasets.downstream_dataset import TennisDownstreamDataset


# -----------------------------
# Helper: device selection
# -----------------------------
def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


# -----------------------------
# Helper: build encoder backbone
# (ResNet-18 with fc removed)
# -----------------------------
def build_resnet18_encoder(device):
    encoder = models.resnet18(weights=None)
    in_dim = encoder.fc.in_features
    encoder.fc = nn.Identity()
    encoder.to(device)
    return encoder, in_dim


# -----------------------------
# Helper: load fine-tuned model
# -----------------------------
def load_model(checkpoint_path, device):
    """
    Loads encoder + linear head from a fine-tuned checkpoint.
    Checkpoints were saved as:
        {"encoder": encoder.state_dict(),
         "head": head.state_dict()}
    """
    # build encoder backbone
    encoder, in_dim = build_resnet18_encoder(device)
    head = nn.Linear(in_dim, 2)   # 2 classes: forehand/backhand
    head.to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    encoder.load_state_dict(ckpt["encoder"])
    head.load_state_dict(ckpt["head"])

    encoder.eval()
    head.eval()

    return encoder, head


# -----------------------------
# Helper: build val loader
# -----------------------------
def get_val_loader(batch_size=32):
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    val_ds = TennisDownstreamDataset(
        root="datasets/data_downstream",
        split="val",
        transform=transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return val_ds, val_loader


# -----------------------------
# Evaluation loop
# -----------------------------
def run_inference(model_name, encoder, head, val_ds, val_loader, device):
    y_true = []
    y_pred = []
    image_indices = []  # to retrieve images later for qualitative plots

    with torch.no_grad():
        for batch_idx, (imgs, labels) in enumerate(val_loader):
            imgs = imgs.to(device)
            labels = labels.to(device)

            feats = encoder(imgs)
            logits = head(feats)
            preds = logits.argmax(dim=1)

            y_true.append(labels.cpu().numpy())
            y_pred.append(preds.cpu().numpy())

            # store global indices of images in val_ds
            start_idx = batch_idx * val_loader.batch_size
            indices = list(range(start_idx, start_idx + labels.size(0)))
            image_indices.extend(indices)

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    acc = (y_true == y_pred).mean()
    print(f"{model_name} accuracy: {acc:.4f} ({(acc*100):.2f}%)")

    return y_true, y_pred, image_indices


# -----------------------------
# Confusion matrix plotting
# -----------------------------
def plot_confusion_matrix(y_true, y_pred, model_name, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["forehand", "backhand"],
    )

    fig, ax = plt.subplots(figsize=(4, 4))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title(f"{model_name} – Confusion Matrix")

    if save_path is not None:
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        print(f"Saved confusion matrix to {save_path}")
    else:
        plt.show()


# -----------------------------
# Qualitative grids
# -----------------------------
def denormalize(tensor):
    """
    Undo ImageNet normalization for visualization.
    tensor: [3, H, W]
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return tensor * std + mean


def plot_examples(val_ds, image_indices, y_true, y_pred,
                  model_name, correct=True, num_examples=8, save_path=None):
    """
    Plot a grid of correctly classified or misclassified examples.
    """
    indices = [
        idx for idx, (yt, yp) in enumerate(zip(y_true, y_pred))
        if (yt == yp) == correct
    ]

    if len(indices) == 0:
        print(f"No {'correct' if correct else 'incorrect'} examples found for {model_name}.")
        return

    # sample up to num_examples
    indices = random.sample(indices, min(num_examples, len(indices)))

    n_cols = 4
    n_rows = int(np.ceil(len(indices) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = np.array(axes).reshape(-1)

    for ax, idx in zip(axes, indices):
        img, label = val_ds[idx]
        img_vis = denormalize(img).clamp(0, 1)  # back to [0,1]

        ax.imshow(np.transpose(img_vis.numpy(), (1, 2, 0)))
        ax.axis("off")

        true_label = "forehand" if y_true[idx] == 0 else "backhand"
        pred_label = "forehand" if y_pred[idx] == 0 else "backhand"

        color = "green" if correct else "red"
        ax.set_title(f"T: {true_label}\nP: {pred_label}", color=color)

    # hide any leftover axes
    for ax in axes[len(indices):]:
        ax.axis("off")

    title = f"{model_name} – {'Correct' if correct else 'Incorrect'} Predictions"
    plt.suptitle(title)

    if save_path is not None:
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(save_path, dpi=200)
        print(f"Saved qualitative grid to {save_path}")
    else:
        plt.show()


# -----------------------------
# Main
# -----------------------------
def main():
    device = get_device()
    print("Using device:", device)

    # 1) Load val dataset + loader
    val_ds, val_loader = get_val_loader(batch_size=32)

    # 2) Evaluate MoCo fine-tuned model
    moco_ckpt = "checkpoints/checkpoints_moco/best_ft_model.pth"
    if os.path.exists(moco_ckpt):
        print("\n=== Evaluating MoCo fine-tuned model ===")
        moco_encoder, moco_head = load_model(moco_ckpt, device)
        moco_y_true, moco_y_pred, moco_img_idx = run_inference(
            "MoCo v2", moco_encoder, moco_head, val_ds, val_loader, device
        )

        plot_confusion_matrix(
            moco_y_true,
            moco_y_pred,
            model_name="MoCo v2",
            save_path="moco_confusion_matrix.png",
        )

        plot_examples(
            val_ds,
            moco_img_idx,
            moco_y_true,
            moco_y_pred,
            model_name="MoCo v2",
            correct=True,
            num_examples=8,
            save_path="moco_correct_examples.png",
        )

        plot_examples(
            val_ds,
            moco_img_idx,
            moco_y_true,
            moco_y_pred,
            model_name="MoCo v2",
            correct=False,
            num_examples=8,
            save_path="moco_incorrect_examples.png",
        )
    else:
        print("MoCo checkpoint not found:", moco_ckpt)

    # 3) Evaluate JePA fine-tuned model
    jepa_ckpt = "checkpoints/checkpoints_ssl/best_ft_model_jepa.pth"
    if os.path.exists(jepa_ckpt):
        print("\n=== Evaluating JePA fine-tuned model ===")
        jepa_encoder, jepa_head = load_model(jepa_ckpt, device)
        jepa_y_true, jepa_y_pred, jepa_img_idx = run_inference(
            "JePA SSL", jepa_encoder, jepa_head, val_ds, val_loader, device
        )

        plot_confusion_matrix(
            jepa_y_true,
            jepa_y_pred,
            model_name="JePA SSL",
            save_path="jepa_confusion_matrix.png",
        )

        plot_examples(
            val_ds,
            jepa_img_idx,
            jepa_y_true,
            jepa_y_pred,
            model_name="JePA SSL",
            correct=True,
            num_examples=8,
            save_path="jepa_correct_examples.png",
        )

        plot_examples(
            val_ds,
            jepa_img_idx,
            jepa_y_true,
            jepa_y_pred,
            model_name="JePA SSL",
            correct=False,
            num_examples=8,
            save_path="jepa_incorrect_examples.png",
        )
    else:
        print("JePA checkpoint not found:", jepa_ckpt)


if __name__ == "__main__":
    main()
