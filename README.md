# Tennis-SSL
Self-Supervised Learning for Tennis Stroke Classification

This repository implements and compares two self-supervised learning (SSL) approaches for tennis stroke representation learning. The goal is to learn features from unlabeled tennis images and then evaluate these features on a downstream forehand/backhand classification task.

The project explores:
- MoCo v2 (contrastive learning)
- A JePA-style multi-crop SSL method using SIGReg (distributional matching)

After SSL pretraining, we evaluate using:
1. Linear probing (encoder frozen)
2. Full fine-tuning (encoder + head trained)

-------------------------------------------------------------------

## Project Structure

### Project Structure

```
tennis-ssl/
│
├── datasets/
│   ├── downstream_dataset.py        # Labeled FH/BH training data
│   └── lejepa_dataset.py            # Unlabeled dataset for SSL
│
├── models/
│   └── moco_model.py                # MoCo v2 encoder + projection head
│
├── transforms/
│   ├── lejepa_transforms.py         # Multi-crop augmentations for JePA
│   └── moco_transforms.py           # Two-crop transforms for MoCo
│
├── scripts/
│   ├── prepare_data.py              # Builds data_ssl/ and data_downstream/
│   ├── train_lejepa.py              # JePA SSL pretraining
│   ├── train_moco.py                # MoCo SSL pretraining
│   ├── train_classifier_moco.py     # Downstream classifier (MoCo)
│   └── train_classifier_lejepa.py   # Downstream classifier (JePA)
│
└── README.md
```

-------------------------------------------------------------------

## Environment Setup

Create and activate the environment:

conda create -n tennis-ssl python=3.10
conda activate tennis-ssl
Install dependencies:

```pip install torch torchvision``` \
```pip install scikit-learn pillow tqdm``` \
```pip install lejepa```

-------------------------------------------------------------------

## Preparing the Data

Place your raw datasets inside:

```datasets/data/
    foreback_raw/
    actions_raw/
```

Then run:

```python scripts/prepare_data.py```

This script generates:

datasets/data_ssl/images/        (unlabeled SSL pool)
datasets/data_downstream/        (labeled FH/BH classification set)

-------------------------------------------------------------------

## SSL Pretraining

### JePA-style SSL (SIGReg loss)

python scripts/train_lejepa.py

This trains a ResNet-18 encoder using multi-crop views and the SIGReg distributional matching loss.

Checkpoints saved to:

checkpoints_ssl/

### MoCo v2 Pretraining

python scripts/train_moco.py

This trains a ResNet-18 encoder with a 2-layer projection head and a momentum encoder.

Checkpoints saved to:

checkpoints_moco/

-------------------------------------------------------------------

## Downstream Evaluation

### Classification with MoCo encoder

python scripts/train_classifier_moco.py

Loads the MoCo checkpoint, strips the projection head, and trains:
1. Linear probe
2. Full fine-tuning (20 epochs)

### Classification with JePA encoder

python scripts/train_classifier_lejepa.py

Same evaluation protocol but using the JePA encoder.

-------------------------------------------------------------------

## Results Summary

Below is the final performance achieved on the downstream FH/BH classification task.

MoCo v2:
- Best fine-tuned accuracy: 92%

JePA-style SSL:
- Best fine-tuned accuracy: 85%

-------------------------------------------------------------------

## Training Script Workflow Explained

Each SSL method is used only for representation learning. The encoder is pretrained on unlabeled images using MoCo or JePA. After SSL pretraining, the projection head (MoCo) or identity head (JePA) is removed, and the encoder is used as a frozen feature extractor for linear probing. Then the entire encoder is fine-tuned end-to-end for improved downstream accuracy.



