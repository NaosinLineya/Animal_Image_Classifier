# Animal Image Classifier: Cat, Dog, and Wild Animals
### Multiclass Image Classification Using a Custom CNN in PyTorch

---

## Overview

This project implements a Convolutional Neural Network (CNN) from scratch using **PyTorch** to classify images into three categories: **cat**, **dog**, and **wild** animals. It demonstrates the full deep learning pipeline — from custom dataset loading and preprocessing to model training, evaluation, and single-image inference — without relying on any pretrained backbone.

> **Scope**: A from-scratch CNN implementation serving as a practical study of convolutional architectures for multiclass image classification.

---

## Dataset

- **Source**: AFHQ (Animal Faces-HQ) dataset
- **Classes**: `cat`, `dog`, `wild` (3-class classification)
- **Input resolution**: Images resized to 128×128 RGB
- **Split**:

| Split | Proportion |
|---|---|
| Training | 70% |
| Validation | 15% |
| Test | 15% |

- **Preprocessing**: Resize to 128×128, convert to float tensor. Labels encoded with `sklearn.LabelEncoder`.

---

## Model Architecture

A custom CNN built with three convolutional blocks followed by fully connected layers:

```
Input: (3, 128, 128)
→ Conv2d(3, 32, 3x3, pad=1)   → MaxPool2d(2,2) → ReLU  →  (32, 64, 64)
→ Conv2d(32, 64, 3x3, pad=1)  → MaxPool2d(2,2) → ReLU  →  (64, 32, 32)
→ Conv2d(64, 128, 3x3, pad=1) → MaxPool2d(2,2) → ReLU  →  (128, 16, 16)
→ Flatten
→ Linear(32768, 128)
→ Linear(128, 3)
Output: class logits (3 classes)
```

- **Loss Function**: CrossEntropyLoss
- **Optimizer**: Adam (`lr = 1e-4`)
- **Batch Size**: 16
- **Epochs**: 10
- **Device**: GPU (CUDA) if available

---

## Results

| Metric | Value |
|---|---|
| Final Training Accuracy | ~99.40% |
| Final Validation Accuracy | ~95.826% |
| Final Testing Accuracy | ~95.99% |



The ~4% train-validation gap indicates mild overfitting, expected given the absence of regularization techniques such as dropout or data augmentation.

---

## Limitations & Future Work

- **Mild overfitting**: No dropout, batch normalization, or data augmentation applied
- **Fixed input resolution**: 128×128 discards fine-grained texture detail present in higher resolutions
- **No test-time augmentation**: Single forward pass inference only
- **Architecture depth**: Three convolutional layers is shallow relative to state-of-the-art models

**Potential improvements**:
- Add dropout and batch normalization between layers
- Apply data augmentation (random flips, color jitter, rotation)
- Compare against a pretrained backbone (ResNet, EfficientNet) via transfer learning
- Evaluate with precision, recall, and F1-score per class, not just accuracy

---

## Requirements

```
torch
torchvision
numpy
pandas
scikit-learn
matplotlib
Pillow
torchsummary
```

Install with:
```bash
pip install torch torchvision numpy pandas scikit-learn matplotlib Pillow torchsummary
```

---

## Usage

1. Clone the repository
2. Download the AFHQ dataset and place it at the path referenced in the notebook
3. Open and run `image_classifier.ipynb` sequentially

For single-image inference, use the `predict_image(img_path)` function at the bottom of the notebook.

---

## Project Structure

```
├── image_classifier.ipynb     # Main notebook
└── README.md
```

> Note: The AFHQ dataset is not included in this repository due to size. Download it from the official source.

---

## Academic Context

This project is the second in a series of progressively complex deep learning implementations, following a tabular binary classifier and preceding a transfer learning study. It focuses on understanding convolutional feature extraction and the complete image classification pipeline using raw PyTorch without high-level wrappers.

---

## Author

*[Naosin Tabassum Lineya]*
*[Jahangirnagar University]*
*[lineyanaosin@gmail.com]*
