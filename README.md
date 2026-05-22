# Cross-View Human Gait Recognition Through Attention Guided Residual Learning

A robust two-stage deep learning framework for **cross-view human gait recognition** using **Attention Guided Residual Learning**.
This project combines **ResNet-50**, **Horizontal Pyramid Mapping (HPM)**, and **Convolutional Block Attention Module (CBAM)** to improve gait recognition performance under varying viewpoints and covariate conditions.

The framework supports multiple gait template representations including:

* **GEI** - Gait Energy Image
* **GMI** - Gait Motion Image
* **MFEI** - Motion Flow Energy Image

The model is evaluated on the widely used CASIA-B Dataset dataset.

---

# Features

* Cross-view gait recognition framework
* Attention-guided residual learning
* ResNet-50 backbone architecture
* CBAM-based feature enhancement
* Horizontal Pyramid Mapping (HPM)
* Multi-template gait representation support
* Robust against viewpoint and covariate variations
* Deep metric learning for feature discrimination
* High recognition accuracy on CASIA-B dataset

---

# Architecture Overview

The proposed framework consists of two major stages:

## Stage 1 - Gait Template Generation

Input gait sequences are converted into template representations:

* GEI
* GMI
* MFEI

These templates capture spatial and temporal gait information.

## Stage 2 - Deep Feature Learning

The generated gait templates are passed through:

1. **ResNet-50 Backbone**
2. **CBAM Attention Module**
3. **Horizontal Pyramid Mapping (HPM)**
4. **Fully Connected Embedding Layer**

The extracted embeddings are used for gait recognition and matching.

---

# Dataset

This work uses the:

* CASIA-B Dataset

The dataset contains gait sequences captured from multiple viewpoints under different walking conditions:

* Normal walking
* Carrying bag
* Wearing coat

---

# Project Structure

```bash
├── dataset/
│   ├── GEI/
│   ├── GMI/
│   └── MFEI/
│
├── models/
│   ├── resnet50_cbam.py
│   ├── hpm.py
│   └── attention_module.py
│
├── training/
│   ├── train.py
│   ├── test.py
│   └── utils.py
│
├── outputs/
│   ├── checkpoints/
│   └── results/
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/your-username/cross-view-gait-recognition.git

cd cross-view-gait-recognition
```

## Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / macOS

```bash
source venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Requirements

```txt
Python >= 3.8
PyTorch
Torchvision
NumPy
OpenCV
Matplotlib
Scikit-learn
Pillow
```

---

# Training

Run the training script:

```bash
python training/train.py
```

### Example Training Parameters

```bash
python training/train.py \
    --dataset CASIA-B \
    --template GEI \
    --epochs 100 \
    --batch_size 32 \
    --lr 0.0001
```

---

# Testing

Evaluate the trained model:

```bash
python training/test.py
```

---

# Model Components

## ResNet-50 Backbone

Used for deep spatial feature extraction from gait templates.

## CBAM (Convolutional Block Attention Module)

Enhances feature representation through:

* Channel Attention
* Spatial Attention

## Horizontal Pyramid Mapping (HPM)

Captures discriminative local gait features at multiple spatial scales.

---

# Supported Gait Representations

| Representation | Description              |
| -------------- | ------------------------ |
| GEI            | Gait Energy Image        |
| GMI            | Gait Motion Image        |
| MFEI           | Motion Flow Energy Image |

---

# Experimental Results

The proposed method demonstrates strong recognition performance across:

* Cross-view variations
* Clothing changes
* Carrying conditions

Example evaluation metrics:

* Accuracy
* Precision
* Recall
* F1-score

---

# Future Improvements

* Transformer-based gait representation learning
* Vision Transformer (ViT) integration
* Real-time gait recognition
* Multi-modal biometric fusion
* Lightweight edge deployment

---

# Applications

* Intelligent surveillance systems
* Person re-identification
* Smart security systems
* Biometric authentication
* Human activity analysis

---

# Citation

If you use this work in your research, please cite:

```bibtex
@article{attention_gait_recognition,
  title={Cross-View Human Gait Recognition Through Attention Guided Residual Learning},
  author={Your Name},
  journal={Journal/Conference Name},
  year={2026}
}
```

---

# License

This project is licensed under the MIT License.

---

# Acknowledgements

* CASIA-B Dataset
* Computer Vision
* Deep Learning




