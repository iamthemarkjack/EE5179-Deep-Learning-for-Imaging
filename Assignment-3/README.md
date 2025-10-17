## Assignment 3 - Autoencoders

This repository contains five Python scripts implementing parts 1–5 of the Autoencoders assignment.

Each part corresponds to a different autoencoder type or experiment, and the scripts are written to be **independent and self-contained**.

---

## Directory Structure

```
.
├── pca_vs_autoencoder.py
├── standard_autoencoder.py
├── sparse_autoencoder.py
├── denoising_autoencoder.py
├── manifold_learning.py
├── outputs/
│   ├── part1/
│   ├── part2/
│   ├── part3/
│   ├── part4/
│   └── part5/
└── README.md
```

All output images and trained model weights will be saved in the respective `outputs/partX` folders.

---

## Requirements

Install dependencies:

```bash
pip install torch torchvision numpy matplotlib scikit-learn
```

---

## How to Run

Each script trains its respective model and saves outputs.

```bash
# Part 1: PCA vs Autoencoder
python pca_vs_autoencoder.py --epochs 10 --batch-size 128

# Part 2: Standard AE (change hidden size to 64, 128, 256)
python standard_autoencoder.py --hidden-size 128 --epochs 10

# Part 3: Sparse AE (L1 regularization)
python sparse_autoencoder.py --hidden-size 128 --l1 1e-3 --epochs 10

# Part 4: Denoising AE
python denoising_autoencoder.py --latent-dim 1 --noise 0.5 --epochs 10

# Part 5: Manifold Learning AE
python manifold_learning.py --epochs 20
```

---

## Answers to Assignment Questions

### **Comparing PCA and Autoencoders**
- PCA with 30 eigenvectors reconstructs digits reasonably but produces smooth, blurry edges.
- The Autoencoder (AE) with 30-dimensional latent achieves **lower reconstruction error (MSE)** and captures **nonlinear features** like stroke variations better.
- AE reconstructions are sharper and visually closer to input digits.

**Conclusion:**  
Autoencoders outperform PCA when learning nonlinear manifolds of digit images, whereas PCA only captures linear variance.

---

### **Standard Autoencoder**
- Hidden sizes tested: 64, 128, 256.
- As the hidden dimension increases:
  - Reconstruction improves.
  - Latent representations become less compressed but more accurate.
- For random noise input: AE outputs blurred images resembling digit-like structures—indicating the model has learned the digit manifold.
- Visualization of encoder weights (filters):
  - The filters appear as **stroke-like patterns**, similar to PCA basis digits.
  - They represent important local stroke components that reconstruct digits.

**Conclusion:**  
Hidden dimension controls compression–reconstruction trade-off. Learned filters resemble digit components, implying the AE learns meaningful substructures.

---

### **Sparse Autoencoders**
- Overcomplete hidden layer (e.g. 1024 units) + L1 sparsity.
- Increasing the L1 penalty enforces **fewer active neurons**, producing:
  - More interpretable, edge-like filters.
  - Slightly higher reconstruction error (if too sparse).
- Average hidden activations for Sparse AE < Standard AE.
- Sparse filters are **cleaner and more localized**, similar to receptive fields in biological neurons.

**Conclusion:**  
Sparsity leads to part-based representations — few neurons activate for any given digit, improving interpretability but at a cost to reconstruction accuracy.

---

### **Denoising Autoencoder**
- Adding Gaussian noise (σ = 0.3, 0.5, 0.8, 0.9):
  - AE trained on noisy inputs can successfully **recover clean digits**.
  - High noise (0.8–0.9) causes partial reconstructions or digit collapse.
- Filters become more **robust and smooth** compared to standard AE filters.
- Standard AE fails on noisy inputs; DAE generalizes better.

**Conclusion:**  
Denoising training encourages AEs to learn noise-invariant features — this improves robustness and generalization.

---

### **Manifold Learning**
- Trained AE: 784 → 64 → 8 → 64 → 784.
- Random noise in pixel space produces unrecognizable images (off the manifold).
- Adding noise in latent (8D) space:
  - Produces smooth transitions between digits.
  - Nearby points decode to visually similar digits.
- The AE’s latent space captures a **low-dimensional manifold of valid digits**.

**Conclusion:**  
AE learns a continuous manifold where small perturbations in latent space correspond to smooth transformations of digit appearance.

---

## Notes
- Each model’s output images and filters are saved to `outputs/partX/`.
- To visualize training progress, check console loss printouts.
- All models use ReLU activations and MSE loss with Adam optimizer.