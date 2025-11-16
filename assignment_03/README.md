# Assignment 03

This assignment contains three tasks on image segmentation and generative models.

---

## Files

- `st124895_task_01.ipynb` - GrabCut segmentation with YOLOv8
- `st124895_task_02.ipynb` - FCN semantic segmentation
- `st124895_task_03.ipynb` - Variational Autoencoder (VAE)

---

## Task 01: GrabCut Segmentation

Uses YOLOv8 to detect persons in images, then applies GrabCut algorithm for segmentation.

**Dataset:** 2 images (asm-1.jpg, asm-2.jpg)

**Libraries:** ultralytics, opencv-python, numpy, matplotlib, pandas

**Method:**
- YOLOv8n for person detection
- GrabCut for foreground extraction
- Tested with 1, 3, and 5 iterations

**Results:**
- Measures foreground pixels, foreground ratio, and IoU between iterations
- asm-1.jpg: High stability (IoU ~0.96-0.99)
- asm-2.jpg: More variation (IoU ~0.74-0.77)

---

## Task 02: FCN Semantic Segmentation

Implements FCN-8s for semantic segmentation on COCO dataset.

**Dataset:** COCO 2017 (20 images, 16 train / 4 test)

**Libraries:** torch, torchvision, pycocotools, numpy, matplotlib, pandas, PIL

**Architecture:**
- Encoder: ResNet50 (pretrained)
- Decoder: Two versions tested
  - Transpose convolution (learnable)
  - Bilinear interpolation (fixed)

**Training:**
- Loss: CrossEntropyLoss
- Optimizer: Adam
- Epochs: 20
- Metrics: Mean IoU, Pixel Accuracy

**Results:**
- Transpose convolution performs slightly better
- Both methods achieve similar reconstruction quality

---

## Task 03: Variational Autoencoder

Trains VAE models on MNIST to learn latent representations and generate new images.

**Dataset:** MNIST (60,000 train / 10,000 test)

**Libraries:** torch, torchvision, numpy, matplotlib, tqdm

**Architecture:**
- Encoder: 4 conv layers → μ and log(σ²)
- Decoder: 4 transpose conv layers
- Loss: BCE (reconstruction) + KL divergence

**Training:**
- Optimizer: Adam (lr=1e-3)
- Epochs: 50 per model
- Two models: latent_dim=128 and latent_dim=256

**Results:**

| Model | Train Loss | Test Loss |
|-------|------------|-----------|
| latent_dim=128 | 93.15 | 93.72 |
| latent_dim=256 | 93.30 | 94.20 |

**Observations:**
- latent_dim=128 generalizes better (lower test loss)
- Both models generate recognizable digits
- Interpolation shows smooth transitions in latent space

---

## Requirements

```bash
# Common
pip install torch torchvision opencv-python matplotlib numpy pandas

# Task 01
pip install ultralytics

# Task 02
pip install pycocotools

# Task 03
pip install tqdm
```

---

## Running the Notebooks

Each notebook runs independently. GPU recommended for tasks 02 and 03.

**Estimated runtime:**
- Task 01: ~2-3 minutes
- Task 02: ~20-30 minutes (with GPU)
- Task 03: ~30-40 minutes (with GPU)

---

Student ID: st124895
