# Assignment 02 - Computer Vision

## Overview

- **Task 1**: CNN Architectures Comparison on CIFAR-10
- **Task 2**: Object Detection Model Comparison (Two-Stage vs Single-Stage)
- **Task 3**: Object Tracking Algorithms Comparison

## Prerequisites

### Recommended Setup

This project is best run using [uv](https://github.com/astral-sh/uv), a fast Python package installer and resolver.

**Install uv:**
```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Alternative Setup (pip)

If you prefer using pip, you can install dependencies using `requirements.txt` or directly from the notebooks.

## Running the Notebooks

### Task 1: CNN Architectures Comparison

**Objective**: Compare SimpleCNN and ResNet18 architectures with different optimizers (SGD vs Adam) on CIFAR-10 dataset.

**File**: `st124895_notebook_task_1.ipynb`

**Key Features**:
- Two CNN architectures (SimpleCNN + ResNet18)
- Optimizer comparison (SGD vs Adam)
- Learning rate schedulers (StepLR, ReduceLROnPlateau)
- Visualizations: training curves, confusion matrices, misclassified images, activation maps
- Performance comparison and analysis

**Run with uv**:
```bash
# Install dependencies
uv pip install torch torchvision matplotlib numpy scikit-learn pandas

# Launch Jupyter
uv run jupyter notebook st124895_notebook_task_1.ipynb
```

**Expected Output**:
- Training/validation accuracy and loss curves
- Confusion matrices for all model combinations
- Activation maps visualization
- Performance comparison table
- Best model: ResNet18 + Adam (~80.25% validation accuracy)

**Runtime**: ~10-15 minutes (with GPU) per model configuration

---

### Task 2: Object Detection Model Comparison

**Objective**: Compare Faster R-CNN (two-stage) and YOLOv8 (single-stage) detectors on COCO 2017 mini dataset.

**File**: `st124895_notebook_task_2.ipynb`

**Key Features**:
- Two-stage detector: Faster R-CNN (ResNet50 backbone)
- Single-stage detector: YOLOv8n
- Metrics: mAP@0.5, mAP@0.5:0.95, FPS, model size, memory usage
- Dataset: COCO 2017 validation (128 images)
- Precision-recall curves
- Side-by-side detection visualizations

**Run with uv**:
```bash
# Install dependencies
uv pip install torch torchvision ultralytics fiftyone pycocotools torchmetrics opencv-python matplotlib pandas tqdm psutil scikit-learn

# Launch Jupyter
uv run jupyter notebook st124895_notebook_task_2.ipynb
```

**Expected Output**:
- Quantitative comparison table (mAP, FPS, memory, model size)
- Precision-recall curves
- Visual detection results (8 sample images)
- Per-class detection summary
- Faster R-CNN: Higher accuracy (mAP@0.5: 72.66%)
- YOLOv8: Faster inference (14.43 FPS) and smaller model (12 MB)

**Runtime**: ~5-10 minutes (dataset download + inference)

---

### Task 3: Object Tracking Algorithms Comparison

**Objective**: Compare CSRT and MOSSE trackers on high-resolution video with comprehensive performance metrics.

**File**: `st124895_notebook_task_3.ipynb`

**Key Features**:
- Two trackers: CSRT (accuracy-focused) vs MOSSE (speed-focused)
- Metrics: Processing FPS, Playback FPS, Tracking Stability, Failure Detection
- Side-by-side real-time comparison
- Adaptive UI with resolution-independent text sizing
- Performance analysis and trade-off discussion

**Run with uv**:
```bash
# Install dependencies
uv pip install opencv-python numpy

# Launch Jupyter
uv run jupyter notebook st124895_notebook_task_3.ipynb
```

**Video Requirements**:
- Video file: `14508384_2160_3840_60fps.mp4` (4K video at 60 FPS)
- Place video in the same directory as the notebook

**Usage Instructions**:
1. Run all cells in the notebook
2. When prompted, select ROI (Region of Interest) by dragging a box on the initial frame
3. Press ENTER to confirm selection
4. Watch the side-by-side tracker comparison
5. Press Q or ESC to stop playback early
6. Review the final metrics comparison and analysis

**Expected Output**:
- Real-time tracker comparison window
- Final metrics:
  - Processing FPS: MOSSE (~74 fps) vs CSRT (~23 fps)
  - Playback FPS: Actual video reproduction speed
  - Tracking Stability: 100% for both (on simple ROI)
  - Failure count and max consecutive failures
- Comprehensive analysis explaining:
  - Why 3 different FPS measurements exist
  - Performance bottlenecks (rendering overhead)
  - Trade-offs between speed and accuracy

**Runtime**: Depends on video length and tracker performance (~8-10 FPS playback for CSRT+MOSSE comparison)

---

## Results Summary

### Task 1: CNN Architectures
| Model | Optimizer | Validation Accuracy | Rank |
|-------|-----------|---------------------|------|
| ResNet18 | Adam | **80.25%** | 1st |
| ResNet18 | SGD | 79.11% | 2nd |
| SimpleCNN | SGD | 75.93% | 3rd |
| SimpleCNN | Adam | 75.41% | 4th |

**Note**: ResNet18 with Adam optimizer achieved the best performance.

### Task 2: Object Detection
| Model | Type | mAP@0.5 | FPS | Model Size |
|-------|------|---------|-----|------------|
| Faster R-CNN | Two-Stage | **72.66%** | 8.80 | 159.79 MB |
| YOLOv8n | Single-Stage | 47.67% | **14.43** | **12.06 MB** |

**Note**: Faster R-CNN wins in accuracy, YOLOv8 wins in speed and efficiency.

### Task 3: Object Tracking
| Tracker | Processing FPS | Algorithm Speed | Stability | Use Case |
|---------|----------------|-----------------|-----------|----------|
| MOSSE | **74.35** | **1093 FPS** | 100% | Real-time applications |
| CSRT | 22.68 | 32.59 FPS | 100% | High-precision tracking |

**Note**: MOSSE is 33x faster in algorithm speed, 3.3x faster in processing. Both achieved perfect stability on the test ROI.


### FiftyOne Dataset Issues (Task 2)
If COCO dataset download fails:
```bash
# Clear cache and retry
rm -rf ~/fiftyone/coco-2017
```

---

## Project Structure
```
assignment_02/
├── README.md                           # This file
├── st124895_notebook_task_1.ipynb     # CNN comparison
├── st124895_notebook_task_2.ipynb     # Object detection comparison
├── st124895_notebook_task_3.ipynb     # Tracker comparison
└── 14508384_2160_3840_60fps.mp4       # Test video (Task 3)
```

---

## Dependencies

### Core Requirements
- Python 3.8+
- PyTorch 2.0+
- torchvision
- OpenCV (cv2)
- NumPy
- Matplotlib

### Task-Specific Requirements
- **Task 1**: scikit-learn, pandas
- **Task 2**: ultralytics, fiftyone, pycocotools, torchmetrics, psutil
- **Task 3**: None (only core requirements)

---

## Author
Student ID: st124895

