# SpinOff

# 🧠 Diffusion Super-Resolution for Low-Dose CT/MRI

### 📍 Overview
This project explores **Diffusion Transformer (DiT)** and **Structured Diffusion (SiT)** models for **medical image super-resolution**.  
The goal is to enhance **low-dose CT and MRI scans**—which are often noisy and blurry—into **high-quality diagnostic images**, while reducing patient radiation exposure or scan time.

We benchmark diffusion-based approaches against conventional deep learning baselines such as **U-Net** and **ESRGAN**, evaluating both **perceptual quality metrics** and **task-driven clinical performance** (e.g., segmentation accuracy).

---

## 🎯 Objectives
- Develop and fine-tune **Diffusion models (DiT/SiT)** for CT/MRI super-resolution.  
- Compare with **U-Net** and **ESRGAN** baselines.  
- Evaluate image quality using:
  - **PSNR (Peak Signal-to-Noise Ratio)**
  - **SSIM (Structural Similarity Index)**
- Assess downstream performance improvements on **segmentation tasks** (e.g., organ or tumor segmentation).

---

## 🧬 Dataset
We use open-source **medical imaging datasets**:
- **Low-Dose CT Challenge Dataset (LDCT, Mayo Clinic)**
- **FastMRI (by NYU + Facebook AI Research)**
- Optional: **BraTS 2021** (for segmentation evaluation)

All datasets are preprocessed to maintain consistent voxel size, slice thickness, and intensity normalization.

---

## ⚙️ Methodology
1. **Preprocessing**
   - Intensity normalization and bias correction  
   - Slice alignment and noise reduction  
   - Downsampling for synthetic low-dose simulation  

2. **Model Training**
   - Implemented DiT/SiT in PyTorch  
   - Baselines: U-Net, ESRGAN  
   - Trained with **L1/L2 + perceptual + diffusion losses**

3. **Evaluation**
   - Quantitative: PSNR, SSIM, LPIPS  
   - Qualitative: Visual comparison of restored slices  
   - Task-driven: Improvement in segmentation accuracy

---

## 🧰 Tech Stack
| Component | Technology |
|------------|-------------|
| Language | Python 3.12 |
| Frameworks | PyTorch, Lightning |
| Visualization | Matplotlib, MONAI, TensorBoard |
| Metrics | scikit-image, lpips |
| Dataset Handling | nibabel, SimpleITK |

---

## 📊 Results (coming soon)
| Model | PSNR ↑ | SSIM ↑ | Segmentation Dice ↑ |
|--------|---------|--------|----------------------|
| U-Net | — | — | — |
| ESRGAN | — | — | — |
| DiT | — | — | — |
| SiT | — | — | — |

*(Results will be updated after training and evaluation.)*

---

## 📁 Repository Structure

├── data/ # Datasets
├── models/ # Model architectures
├── notebooks/ # Experiment notebooks
├── scripts/ # Preprocessing & training scripts
├── results/ # Output images and metrics
├── README.md # Project overview
└── requirements.txt # Dependencies
