# X-Ray Image Denoising Pipeline with PyTorch

A modular Python/PyTorch project for evaluating denoising methods on chest X-ray images.

## Overview

This project builds a complete denoising workflow for X-ray images using:

- **Dataset:** PneumoniaMNIST
- **Noise models:** Gaussian and Poisson
- **Denoising methods:** BM3D and Wavelet denoising
- **Evaluation metrics:** MSE, MAE, NRMSE, PSNR, SSIM
- **Extra analysis:** runtime per image and RGB vs grayscale comparison

The project automatically downloads the dataset, applies controlled noise, performs denoising, computes quantitative metrics, and saves publication-style figures and CSV summaries.

---

## Main Features

- Automatic dataset download with MedMNIST
- Deterministic experiment design using a fixed random seed
- Separate Gaussian and Poisson experiments
- Comparison of:
  - noisy vs denoised
  - BM3D vs Wavelet
  - RGB vs grayscale
- Metric summary tables
- Runtime tracking
- Clean plots and dashboards

---

## Dataset

This repository uses **PneumoniaMNIST**.

Important note:
- the original dataset is grayscale chest X-ray data
- the **RGB branch** is created by repeating the grayscale image into 3 channels
- this enables a controlled RGB-vs-gray processing comparison

Default experiment settings:
- image size: `224 x 224`
- split: `test`
- number of images: `40`

---

## Methods

### Noise models
- Gaussian noise
- Poisson noise

### Denoising methods
- BM3D
- Wavelet denoising

### Metrics
- MSE
- MAE
- NRMSE
- PSNR
- SSIM
- runtime per image

---

## Project Structure

```text
image-denoising-pipeline/
│
├── docs/
│   └── figures/
├── src/
│   ├── config.py
│   ├── dataset_loader.py
│   ├── image_utils.py
│   ├── noise.py
│   ├── denoise.py
│   ├── metrics.py
│   ├── visualization.py
│   └── main.py
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt