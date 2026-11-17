# Truck CCTV ROI Classifier (ResNet50)

High-resolution video inference pipeline for CCTV truck status classification (`empty` vs `full`) using a fine-tuned ResNet50 model on dynamic Region of Interest (ROI).

---

## Overview

- **Architecture:** ResNet50 with custom Dropout (0.05) & Linear FC classification head (2 classes: `empty`, `full`).
- **Model Checkpoint:** `model/best_validation_resnet50.pth`.
- **Automatic ROI Scaling:** Automatically scales the original training ROI from native CCTV training resolution (704 x 576) to high-resolution inference video (e.g., 2560 x 1440).
- **Uncertainty Rejection:** Predictions below the confidence threshold (0.70) are flagged as `UNCERTAIN` on the video feed while logging raw probabilities.
- **Outputs:**
  1. Annotated MP4 video with real-time prediction overlays, confidence, and timestamp.
  2. Per-frame prediction CSV file.
  3. Comprehensive execution summary JSON file.

---

## Directory Structure

```text
truck_detection/
├── model/
│   └── best_validation_resnet50.pth   # Model weights (tracked via Git LFS)
├── main.py                            # Video inference pipeline script
├── roi.json                           # Training and scaled ROI coordinates & config
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git exclusion rules
├── .gitattributes                     # Git LFS definitions
└── README.md                          # Documentation
```

---

## ROI Configuration (`roi.json`)

| Parameter | Training Video | Scaled Inference Video (2560x1440) |
| :--- | :--- | :--- |
| **Resolution** | `704 x 576` | `2560 x 1440` |
| **X Scale Factor** | 1.0 | 3.636364 |
| **Y Scale Factor** | 1.0 | 2.500000 |
| **ROI Box (X1, Y1, X2, Y2)** | `[146, 4, 445, 273]` | `[531, 10, 1618, 682]` |
| **ROI Dimensions (W x H)** | `299 x 269` | `1087 x 672` |

---

## Installation

Ensure Python 3.8+ is installed. Clone the repository and install dependencies:

```bash
git clone https://github.com/amrendrayadav-ux/truck_detection.git
cd truck_detection
git lfs pull
pip install -r requirements.txt
```

---

## Usage

### Local Execution
```bash
python main.py
```

### Google Colab
The script is single-cell ready. When run in Colab, it mounts Google Drive and looks for video/model checkpoints seamlessly:
```python
# In Google Colab
%run main.py
```
During execution, you can specify custom start and end timestamps (e.g. `00:00:00` or `01:30:00`), or press Enter to process the entire video.
