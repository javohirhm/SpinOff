# Diffusion Super-Resolution for Low-Dose CT/MRI

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)  
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)  
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)  
[![Open Bicubic Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1v77SN-VDLmxrJjmDwiikr-s3xILpIm9B?usp=sharing)

Official implementation for **“Diffusion Super-Resolution for Low-Dose CT/MRI”**,  
Computer Vision Course Project (Fall 2025).

---

## 📋 Project Overview

This repository implements and compares two super-resolution approaches for medical image enhancement:

- **Bicubic Interpolation** — classical non-learned baseline  
- **SR3** — diffusion-based super-resolution model

The goal is to reconstruct high-quality CT/MRI images from simulated low-dose inputs using the IXI MRI dataset.

---

## 📂 Dataset

- **Original IXI Dataset (3D volumes)**  
  https://brain-development.org/ixi-dataset/

- **Preprocessed IXI Slices (3D → 2D)**  
  https://drive.google.com/drive/folders/1tvy2f7bHvSRiuRfDd0bX7sIp09xb7q0u?usp=sharing

---

## 🧠 SR3 Architecture Overview

```
LR (128×128)
   ↓ Learned Upsampling (Conv → SiLU → ConvTranspose → SiLU → Conv)
LR↑ (256×256)
   ↓
Concat([x_t, LR↑])
   ↓
U-Net with:
  • Residual blocks (GroupNorm → SiLU → Conv)
  • Time embeddings
  • Self-attention at 16×16 resolution
   ↓
Predicted noise ε̂
```

---

## 🧪 Google Colab Notebooks

### **1. Bicubic Baseline — Training + Results (Full Pipeline)**  
All bicubic training, evaluation, and visual outputs are available here:  
👉 https://colab.research.google.com/drive/1v77SN-VDLmxrJjmDwiikr-s3xILpIm9B?usp=sharing

This notebook performs:

- LR generation  
- 2× bicubic super-resolution  
- PSNR / SSIM evaluation  
- HR vs LR vs Bicubic comparisons  

---

### **2. SR3 Sampling Notebook (SR3-FiLM removed)**  
Sampling experiments for SR3 model and early SR3-FiLM attempts.  
SR3-FiLM failed and is **not included in the final report**.

👉 https://colab.research.google.com/drive/1Hq7EfXlEdcQ_HpTVwQQalRJfOZBSrkp8#scrollTo=Bb77WrzCwPn5

Includes:

- SR3 forward diffusion  
- Reverse sampling (1000 steps)  
- Visual SR samples  
- FiLM tests (discarded in final project)  

---

## 📈 Results Comparison

| Method       | PSNR (↑) | SSIM (↑) | Notes |
|--------------|----------|----------|-------|
| Bicubic      | 29.93 dB | 0.7778   | Smooth, stable baseline |
| SR3 (ours)   | 19.53 dB | 0.2630   | Underfitting due to limited training |

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/javohirhm/SpinOff.git
cd SpinOff
pip install -r requirements.txt
```

---

## 🧪 Training SR3

```bash
python train_sr3.py --epochs 20 --batch_size 4 --lr 2e-5
```

---

## 🔍 Evaluation

```bash
python evaluate.py --model sr3 --testset ./data/ixi/test
```

Outputs include:

- PSNR  
- SSIM  
- Reconstructed images  

---

## 📦 Pretrained Models & Outputs

Download trained SR3 checkpoints and sample outputs:  
https://drive.google.com/drive/folders/1oaR17lrwzlEUqSmqazDStAzBkD8pHdqF?usp=sharing

---

## 📁 Project Structure

```
SpinOff/
│── bicubic/
│── sr3/
│── utils/
│── data/
│── train_sr3.py
│── evaluate.py
│── requirements.txt
└── README.md
```

---

## 🤝 Contributors

- **Gulrukhsor Akhmadjanova**  
- **Javokhir Khoshimov**

---

## 📜 License

Released under the **MIT License**.

---

## ⭐ Acknowledgements

Built using the IXI Dataset and inspired by SR3 (Saharia et al., 2021) and DDPM (Ho et al., 2020).
