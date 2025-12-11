# Evaluating Self-Supervised Learning and YOLO Models as Foundations for Automated Match Commentary

This repository implements and compares multiple approaches for tennis stroke recognition, including two self-supervised learning (SSL) models (MoCo v2 and JePA), a supervised ResNet baseline, and a YOLOv3 object-detection model. The broader goal is to build components toward automated tennis commentary, requiring player and racket detection, stroke recognition, temporal reasoning over video, natural language generation (LLMs). 

-------------------------------------------------------------------

## Project Structure

### Project Structure

```
## Project Structure

```text

├── datasets/
│   ├── downstream_dataset.py        # Labeled FH/BH dataset loader
│   └── lejepa_dataset.py            # Unlabeled dataset loader + JePA augmentations
│
├── figures/                         # Visualizations and qualitative examples
│   ├── jepa_confusion_matrix.png
│   ├── jepa_correct_examples.png
│   ├── jepa_incorrect_examples.png
│   ├── moco_confusion_matrix.png
│   ├── moco_correct_examples.png
│   ├── moco_incorrect_examples.png
│   └── supervised_confusion_matrix.png
│
├── models/
│   └── moco_model.py                # MoCo v2 encoder implementation
│
├── scripts/
│   ├── baseline_supervised_resnet.py   # ResNet baseline training
│   ├── eval_results.py                 # Metrics + confusion matrices
│   ├── lejepa_transforms.py            # JePA augmentation pipeline
│   ├── moco_transforms.py              # MoCo augmentation pipeline
│   ├── prepare_data.py                 # Train/val split generation
│   ├── train_classifier_lejepa.py      # Linear probe & finetuning for JePA
│   ├── train_classifier_moco.py        # Linear probe & finetuning for MoCo
│   ├── train_lejepa.py                 # JePA pretraining loop
│   └── train_moco.py                   # MoCo v2 pretraining loop
│
├── YOLOv3/                             # YOLO training code & configs
│   └── ...
│
├── .gitignore
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






