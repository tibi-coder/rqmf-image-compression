# Regularized Quantized Matrix Factorization (RQMF) for Image Compression

This repository contains an implementation of **Regularized Quantized Matrix Factorization (RQMF)** for lossy image compression.

The method approximates images using a low-rank matrix factorization while constraining the factor matrices to **bounded integer values**. By incorporating quantization directly into the optimization process, the algorithm accounts for quantization effects during factorization instead of applying quantization as a separate step.

This implementation accompanies research work that will appear at the **IFAC World Congress 2026**.

---

## Method Overview

Given an image matrix **Y**, the compression method computes a low-rank approximation:

**Y ≈ U Vᵀ**

where:

* **U ∈ ℤ^(m × d)**
* **V ∈ ℤ^(n × d)**

are integer-valued factor matrices constrained to the interval **[α, β]**.

The optimization problem is solved using a **cyclic coordinate minimization algorithm** that iteratively updates the columns of the factor matrices.
Regularization terms are introduced to improve stability and compression efficiency during the optimization process.

---

## Compression Pipeline

The implemented compression pipeline follows these steps:

1. Convert RGB images to **YCbCr color space**
2. Downsample chrominance channels
3. Divide image channels into non-overlapping patches
4. Apply **regularized quantized matrix factorization**
5. Store compressed factor matrices
6. Reconstruct the image from the quantized representation

---

## Datasets

The experiments are performed on commonly used benchmark datasets for image compression research.

### Kodak Dataset

The **Kodak dataset** contains **24 high-quality lossless images** with a resolution of **768 × 512** pixels.
It is one of the most widely used datasets for evaluating classical image compression algorithms and rate–distortion performance.

The images cover a diverse range of scenes including natural landscapes, objects, and indoor environments, making the dataset suitable for evaluating compression algorithms across different visual characteristics.

### CLIC Dataset

The **CLIC (Challenge on Learned Image Compression) dataset** contains high-resolution images designed for evaluating modern image compression techniques.

In the experiments we use a subset of **30 high-quality images**, which provide a variety of textures, structures, and lighting conditions.
Compared to Kodak, CLIC images often contain more complex visual structures, making them useful for testing compression methods under more challenging scenarios.

---

## Evaluation

Compression performance is evaluated using standard **rate–distortion metrics**:

* **PSNR** — Peak Signal-to-Noise Ratio
* **SSIM** — Structural Similarity Index
* **BPP** — Bits per Pixel

The script evaluates these metrics for multiple compression ratios and stores the results in a CSV file.

---

## Usage

Place images inside an `images/` directory and run:

```
python compression_qmf.py
```

The script will:

* compress the images
* reconstruct them
* compute PSNR, SSIM and BPP
* save the results in `rqmf_results.csv`

---

## Requirements

Install the dependencies with:

```
pip install -r requirements.txt
```

---

Automatic Control and Systems Engineering Department
University Politehnica of Bucharest
