# Computer Vision Assignment 01

## Quick Start

### 1. Clone and navigate to the project directory
```bash
git clone https://github.com/LmedinaJ/cv_course.git
cd cv_course/assignment_01
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the application
```bash
python app_allone.py
```

### 4. Use the application
- Press **'q'** to quit
- Use keys **1-9** to switch between modes
- Mode-specific controls appear on screen

## Modes

| Key | Mode | Description |
|-----|------|-------------|
| **1** | Normal | Camera without processing |
| **2** | Color Conversion | RGB, Grayscale, HSV |
| **3** | Contrast/Brightness | Adjustment with sliders |
| **4** | Histogram | RGB histogram visualization |
| **5** | Transformation | Translation, rotation, scaling |
| **6** | Calibration | Camera calibration with chessboard |
| **7** | Filters | Blur, Canny, Bilateral, Hough Lines |
| **8** | Panorama | Capture and create panoramas |
| **9** | AR | Augmented reality with 3D model |

## Dependencies

- **OpenCV** (`cv2`) >= 4.8.0
- **NumPy** >= 1.24.0
- **Python 3.x**

Install with: `pip install -r requirements.txt`


## Note
This application is contained in a single file `app_allone.py` for easy distribution and execution.