# Evaluating Self-Supervised Learning and YOLO Models as Foundations for Automated Match Commentary

This repository implements and compares multiple approaches for tennis stroke recognition, including two self-supervised learning (SSL) models (MoCo v2 and JePA), a supervised ResNet baseline, and a YOLOv3 object-detection model. The broader goal is to build components toward automated tennis commentary, requiring player and racket detection, stroke recognition, temporal reasoning over video, natural language generation (LLMs). 

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

```
conda create -n tennis-ssl python=3.10
conda activate tennis-ssl
```

Install dependencies:

```
pip install torch torchvision
pip install scikit-learn pillow tqdm
pip install lejepa
```
-------------------------------------------------------------------

## Preparing the Data

Your datasets should already exist on disk in two parts: 
- Unlabeled SSL dataset
- Labeled downstream dataset

The dataset loaders in: 

```
datasets/downstream_dataset.py
datasets/lejepa_dataset.py
```

Expect the following structure after you run the script below: 

```
datasets/
    data_ssl/images/           # Unlabeled images for SSL
    data_downstream/           # forehand/ + backhand/ folders
```

Run the preprocessig script to automatically populate these directories: 

```
python scripts/prepare_data.py

```


-------------------------------------------------------------------

## SSL Models 

### Model Pretraining

#### JePA-style SSL

```
python scripts/train_lejepa.py
```

This trains a ResNet-18 encoder using multi-crop views and the SIGReg distributional matching loss.

Checkpoints saved to:

```
checkpoints_ssl/
```

#### MoCo v2 Pretraining

```
python scripts/train_moco.py
```

This trains a ResNet-18 encoder with a 2-layer projection head and a momentum encoder.

Checkpoints saved to:

```
checkpoints_moco/
```

### Downstream Evaluation

#### Classification with MoCo encoder

```
python scripts/train_classifier_moco.py
```

Loads the MoCo checkpoint, strips the projection head, and trains:
1. Linear probe
2. Full fine-tuning (20 epochs)

#### Classification with JePA encoder

```
python scripts/train_classifier_lejepa.py
```

Same evaluation protocol but using the MoCo encoder.

-------------------------------------------------------------------

## ResNet Model  

To benchmark against a standard supervised model, we train ResNet-18 only on the labeled forehand/backhand dataset.

Run: 

```
python scripts/baseline_supervised_resnet.py
```

This script: 
- loads the labeled dataset from datasets/data_downstream/
- trains a scratch ResNet-18
- Evaluates on the same validation split

-------------------------------------------------------------------

## YOLO Model 

-------------------------------------------------------------------

## Final Results 

Below are the final-stroke classification accuracies: 
1. ResNet: 95%
2. MoCo v2: 92%
3. JePA: 85%
4. YOLO v3: 67%




