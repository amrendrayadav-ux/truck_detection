# ======================================================================
# RESNET50 CCTV ROI CLASSIFIER — HIGH RESOLUTION VIDEO INFERENCE
# COMPLETE SINGLE-CELL GOOGLE COLAB & LOCAL CODE
# ======================================================================
#
# TRAINING VIDEO RESOLUTION: 704 x 576
# INFERENCE VIDEO RESOLUTION: 2560 x 1440
#
# ORIGINAL TRAINING ROI:
# X1=146, Y1=4, X2=445, Y2=273
#
# ROI IS AUTOMATICALLY SCALED TO THE INFERENCE VIDEO.
#
# MODEL:
# best_validation_resnet50.pth
#
# CLASSES:
# empty = 0
# full  = 1
#
# OUTPUT:
# 1. Annotated MP4 video
# 2. Per-frame prediction CSV
# 3. Summary JSON
# ======================================================================


# ========================= 1. IMPORTS ================================

import os
import cv2
import json
import time
import math
import torch
import numpy as np
import pandas as pd

from PIL import Image
from datetime import timedelta
from torchvision import models, transforms
try:
    from tqdm.notebook import tqdm
except ImportError:
    from tqdm import tqdm


# ========================= 2. GOOGLE DRIVE ============================

try:
    from google.colab import drive
    drive.mount('/content/drive')
except ImportError:
    pass


# ========================= 3. CONFIGURATION ===========================

# Default input video path (Colab Drive or local)
VIDEO_PATH = (
    "/content/drive/MyDrive/Amrendra_Guardex/tin_counter/"
    "input_videos/Cam_3_Monday_2026-09-21_13-57-42.mp4"
)

# Resolve model checkpoint: local repository model directory if available, fallback to Drive
_REPO_MODEL = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "model",
    "best_validation_resnet50.pth"
)
_DRIVE_MODEL = (
    "/content/drive/MyDrive/Amrendra_Guardex/tin_counter_new_angle/"
    "truck_roi_crops/resnet50_cctv_overfit/"
    "best_validation_resnet50.pth"
)

MODEL_PATH = _REPO_MODEL if os.path.isfile(_REPO_MODEL) else _DRIVE_MODEL

# Model class order
CLASS_NAMES = ["empty", "full"]

# Same model architecture as training
DROPOUT_RATE = 0.05

# Same input size as training
IMAGE_SIZE = 224

# Predictions below this confidence are displayed as UNCERTAIN.
# The actual predicted class and probabilities are still saved.
UNCERTAINTY_THRESHOLD = 0.70

# ----------------------------------------------------------------------
# ORIGINAL TRAINING VIDEO RESOLUTION
# ----------------------------------------------------------------------

TRAIN_WIDTH = 704
TRAIN_HEIGHT = 576

# ----------------------------------------------------------------------
# ORIGINAL TRAINING ROI
# ----------------------------------------------------------------------

TRAIN_X1 = 146
TRAIN_Y1 = 4
TRAIN_X2 = 445
TRAIN_Y2 = 273

# Load from roi.json if present
ROI_JSON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "roi.json")
if os.path.isfile(ROI_JSON_PATH):
    try:
        with open(ROI_JSON_PATH, "r") as f:
            roi_cfg = json.load(f)
        TRAIN_WIDTH = roi_cfg.get("training_resolution", {}).get("width", TRAIN_WIDTH)
        TRAIN_HEIGHT = roi_cfg.get("training_resolution", {}).get("height", TRAIN_HEIGHT)
        orig_roi = roi_cfg.get("original_training_roi", {})
        TRAIN_X1 = orig_roi.get("x1", TRAIN_X1)
        TRAIN_Y1 = orig_roi.get("y1", TRAIN_Y1)
        TRAIN_X2 = orig_roi.get("x2", TRAIN_X2)
        TRAIN_Y2 = orig_roi.get("y2", TRAIN_Y2)
    except Exception as e:
        print(f"Note: Could not load parameters from roi.json: {e}")

# ----------------------------------------------------------------------
# DEVICE
# ----------------------------------------------------------------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 75)
print("RESNET50 CCTV ROI INFERENCE")
print("=" * 75)

print("Device:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ========================= 4. CHECK PATHS =============================

if not os.path.isfile(VIDEO_PATH):
    raise FileNotFoundError(
        f"Input video not found:\n{VIDEO_PATH}"
    )

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(
        f"Model checkpoint not found:\n{MODEL_PATH}"
    )

print("\nInput video:", VIDEO_PATH)
print("Model:", MODEL_PATH)


# ========================= 5. OPEN VIDEO ==============================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError("Could not open input video.")

VIDEO_FPS = cap.get(cv2.CAP_PROP_FPS)

TOTAL_FRAMES = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)

FRAME_WIDTH = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

FRAME_HEIGHT = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

if VIDEO_FPS <= 0:
    cap.release()
    raise RuntimeError("Invalid video FPS.")

VIDEO_DURATION = TOTAL_FRAMES / VIDEO_FPS

print("\n" + "=" * 75)
print("VIDEO INFORMATION")
print("=" * 75)

print(f"Resolution : {FRAME_WIDTH} x {FRAME_HEIGHT}")
print(f"FPS        : {VIDEO_FPS:.3f}")
print(f"Total frames: {TOTAL_FRAMES}")
print(f"Duration   : {VIDEO_DURATION:.2f} seconds")


# ========================= 6. AUTOMATIC ROI SCALING ====================

# Calculate independent X and Y scaling factors.
SCALE_X = FRAME_WIDTH / TRAIN_WIDTH
SCALE_Y = FRAME_HEIGHT / TRAIN_HEIGHT

# Scale the original ROI to the new video.
X1 = int(round(TRAIN_X1 * SCALE_X))
Y1 = int(round(TRAIN_Y1 * SCALE_Y))

X2 = int(round(TRAIN_X2 * SCALE_X))
Y2 = int(round(TRAIN_Y2 * SCALE_Y))

# Validate ROI against actual frame dimensions.
if not (
    0 <= X1 < X2 <= FRAME_WIDTH
    and 0 <= Y1 < Y2 <= FRAME_HEIGHT
):
    cap.release()
    raise ValueError(
        f"Scaled ROI is outside the video frame: "
        f"{X1},{Y1},{X2},{Y2}"
    )

print("\n" + "=" * 75)
print("AUTOMATIC ROI SCALING")
print("=" * 75)

print("Training resolution:", TRAIN_WIDTH, "x", TRAIN_HEIGHT)
print("Inference resolution:", FRAME_WIDTH, "x", FRAME_HEIGHT)

print(f"Scale X: {SCALE_X:.6f}")
print(f"Scale Y: {SCALE_Y:.6f}")

print("\nOriginal ROI:")
print(
    f"X1={TRAIN_X1}, Y1={TRAIN_Y1}, "
    f"X2={TRAIN_X2}, Y2={TRAIN_Y2}"
)

print("\nScaled ROI:")
print(
    f"X1={X1}, Y1={Y1}, "
    f"X2={X2}, Y2={Y2}"
)

print(
    f"ROI width: {X2-X1}, "
    f"ROI height: {Y2-Y1}"
)


# ========================= 7. TIME INPUT ==============================

def parse_time(value):

    value = str(value).strip()

    if not value:
        raise ValueError("Time cannot be empty.")

    if ":" not in value:
        seconds = float(value)

    else:

        parts = value.split(":")

        if len(parts) == 3:

            hours, minutes, secs = parts

            seconds = (
                int(hours) * 3600
                + int(minutes) * 60
                + float(secs)
            )

        elif len(parts) == 2:

            minutes, secs = parts

            seconds = (
                int(minutes) * 60
                + float(secs)
            )

        else:
            raise ValueError(
                "Use HH:MM:SS, MM:SS or seconds."
            )

    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError(
            "Time must be finite and non-negative."
        )

    return seconds


print("\n" + "=" * 75)
print("SELECT VIDEO TIME RANGE")
print("=" * 75)

while True:

    try:

        start_input = input(
            "\nEnter START time (default 00:00:00): "
        ).strip()

        if not start_input:
            start_input = "00:00:00"

        START_TIME = parse_time(start_input)

        if START_TIME >= VIDEO_DURATION:
            raise ValueError(
                "Start time must be inside the video."
            )

        break

    except Exception as e:
        print("Invalid start time:", e)


while True:

    try:

        end_input = input(
            "Enter END time (blank = complete video end): "
        ).strip()

        if not end_input:
            END_TIME = VIDEO_DURATION

        else:
            END_TIME = parse_time(end_input)

        if END_TIME > VIDEO_DURATION:
            raise ValueError(
                "End time exceeds video duration."
            )

        if END_TIME <= START_TIME:
            raise ValueError(
                "End time must be greater than start time."
            )

        break

    except Exception as e:
        print("Invalid end time:", e)


START_FRAME = int(
    round(START_TIME * VIDEO_FPS)
)

END_FRAME = int(
    round(END_TIME * VIDEO_FPS)
)

START_FRAME = max(
    0, min(START_FRAME, TOTAL_FRAMES)
)

END_FRAME = max(
    0, min(END_FRAME, TOTAL_FRAMES)
)

if END_FRAME <= START_FRAME:
    cap.release()
    raise RuntimeError(
        "Selected time range contains no frames."
    )

FRAMES_TO_PROCESS = END_FRAME - START_FRAME

print("\nSelected range:")
print("Start time:", START_TIME)
print("End time:", END_TIME)
print("Start frame:", START_FRAME)
print("End frame (exclusive):", END_FRAME)
print("Frames to process:", FRAMES_TO_PROCESS)


# ========================= 8. LOAD MODEL ==============================

print("\n" + "=" * 75)
print("LOADING RESNET50 CHECKPOINT")
print("=" * 75)

try:

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

except TypeError:

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

if not isinstance(checkpoint, dict):
    raise RuntimeError(
        "Unexpected checkpoint format."
    )

if "model_state_dict" not in checkpoint:
    raise KeyError(
        "Checkpoint does not contain model_state_dict."
    )

# Validate class order if saved in checkpoint.
saved_class_names = checkpoint.get("class_names")

if saved_class_names is not None:

    if list(saved_class_names) != CLASS_NAMES:

        raise ValueError(
            f"Checkpoint class names mismatch: "
            f"{saved_class_names}"
        )

# Validate class mapping if saved.
saved_mapping = checkpoint.get("class_to_idx")

if saved_mapping is not None:

    expected_mapping = {
        "empty": 0,
        "full": 1
    }

    if saved_mapping != expected_mapping:

        raise ValueError(
            f"Checkpoint class mapping mismatch: "
            f"{saved_mapping}"
        )

# Build exactly the same architecture as training.
model = models.resnet50(weights=None)

model.fc = torch.nn.Sequential(
    torch.nn.Dropout(DROPOUT_RATE),
    torch.nn.Linear(
        model.fc.in_features,
        2
    )
)

model.load_state_dict(
    checkpoint["model_state_dict"],
    strict=True
)

model = model.to(DEVICE)

model.eval()

print("Model loaded successfully.")
print("Classes:", CLASS_NAMES)


# ========================= 9. PREPROCESSING ===========================

# Same ImageNet normalization as training.
inference_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

])


# ========================= 10. OUTPUT PATHS ===========================

VIDEO_DIR = os.path.dirname(VIDEO_PATH)

if os.path.isdir(VIDEO_DIR):
    OUTPUT_DIR = os.path.join(
        VIDEO_DIR,
        "resnet50_roi_inference"
    )
else:
    OUTPUT_DIR = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "resnet50_roi_inference"
    )

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

VIDEO_BASENAME = os.path.splitext(
    os.path.basename(VIDEO_PATH)
)[0]

RUN_TIMESTAMP = time.strftime(
    "%Y%m%d_%H%M%S"
)

OUTPUT_VIDEO_PATH = os.path.join(
    OUTPUT_DIR,
    f"{VIDEO_BASENAME}_resnet50_annotated_{RUN_TIMESTAMP}.mp4"
)

OUTPUT_CSV_PATH = os.path.join(
    OUTPUT_DIR,
    f"{VIDEO_BASENAME}_predictions_{RUN_TIMESTAMP}.csv"
)

OUTPUT_JSON_PATH = os.path.join(
    OUTPUT_DIR,
    f"{VIDEO_BASENAME}_summary_{RUN_TIMESTAMP}.json"
)

print("\nOutput directory:")
print(OUTPUT_DIR)


# ========================= 11. VIDEO WRITER ===========================

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)

writer = cv2.VideoWriter(
    OUTPUT_VIDEO_PATH,
    fourcc,
    VIDEO_FPS,
    (FRAME_WIDTH, FRAME_HEIGHT)
)

if not writer.isOpened():

    cap.release()

    raise RuntimeError(
        "Could not create output video."
    )


# ========================= 12. SEEK TO START ==========================

cap.set(
    cv2.CAP_PROP_POS_FRAMES,
    START_FRAME
)

print("\nStarting from frame:", START_FRAME)


# ========================= 13. INFERENCE ===============================

results = []

prediction_counts = {
    "empty": 0,
    "full": 0,
    "uncertain": 0
}

total_inference_time = 0.0
processed_frames = 0

inference_start_time = time.time()

print("\n" + "=" * 75)
print("STARTING INFERENCE")
print("=" * 75)

progress = tqdm(
    total=FRAMES_TO_PROCESS,
    desc="ResNet50 ROI inference",
    unit="frame"
)

try:

    with torch.inference_mode():

        while processed_frames < FRAMES_TO_PROCESS:

            ret, frame = cap.read()

            if not ret:

                print(
                    "\nVideo ended before selected end time."
                )

                break

            original_frame_number = (
                START_FRAME + processed_frames
            )

            frame_timestamp = (
                original_frame_number / VIDEO_FPS
            )

            # ----------------------------------------------------------
            # CROP ONLY THE SCALED ROI
            # ----------------------------------------------------------

            roi_crop = frame[
                Y1:Y2,
                X1:X2
            ]

            if roi_crop.size == 0:

                raise RuntimeError(
                    f"Empty ROI at frame {original_frame_number}"
                )

            # OpenCV BGR -> RGB.
            roi_rgb = cv2.cvtColor(
                roi_crop,
                cv2.COLOR_BGR2RGB
            )

            roi_pil = Image.fromarray(
                roi_rgb
            )

            # Resize and normalize.
            input_tensor = inference_transform(
                roi_pil
            ).unsqueeze(0).to(DEVICE)

            # ----------------------------------------------------------
            # RESNET50 INFERENCE
            # ----------------------------------------------------------

            inference_t0 = time.time()

            logits = model(
                input_tensor
            )

            probabilities = torch.softmax(
                logits,
                dim=1
            )[0]

            if DEVICE.type == "cuda":
                torch.cuda.synchronize()

            inference_time = (
                time.time() - inference_t0
            )

            total_inference_time += inference_time

            probabilities_np = (
                probabilities.cpu().numpy()
            )

            predicted_class_id = int(
                np.argmax(probabilities_np)
            )

            predicted_class = (
                CLASS_NAMES[predicted_class_id]
            )

            confidence = float(
                probabilities_np[predicted_class_id]
            )

            empty_probability = float(
                probabilities_np[0]
            )

            full_probability = float(
                probabilities_np[1]
            )

            # ----------------------------------------------------------
            # DISPLAY CLASSIFICATION
            # ----------------------------------------------------------

            if confidence < UNCERTAINTY_THRESHOLD:

                display_label = "UNCERTAIN"

                prediction_counts["uncertain"] += 1

                display_color = (0, 165, 255)

            else:

                display_label = predicted_class.upper()

                prediction_counts[
                    predicted_class
                ] += 1

                if predicted_class == "full":

                    display_color = (0, 255, 0)

                else:

                    display_color = (0, 0, 255)

            # ----------------------------------------------------------
            # ANNOTATE ORIGINAL FRAME
            # ----------------------------------------------------------

            annotated = frame.copy()

            # Draw scaled ROI.
            cv2.rectangle(
                annotated,
                (X1, Y1),
                (X2, Y2),
                display_color,
                3
            )

            # Prediction panel.
            panel_height = 90

            panel_y1 = max(
                0,
                Y1 - panel_height
            )

            panel_y2 = Y1

            # If there is insufficient space above ROI,
            # put the panel inside/below the ROI.
            if panel_y2 - panel_y1 < 70:

                panel_y1 = Y1

                panel_y2 = min(
                    FRAME_HEIGHT,
                    Y1 + panel_height
                )

            panel_x1 = X1

            panel_x2 = min(
                FRAME_WIDTH,
                X1 + 430
            )

            cv2.rectangle(
                annotated,
                (panel_x1, panel_y1),
                (panel_x2, panel_y2),
                (25, 25, 25),
                -1
            )

            cv2.putText(
                annotated,
                f"Prediction: {display_label}",
                (panel_x1 + 10, panel_y1 + 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                display_color,
                2,
                cv2.LINE_AA
            )

            cv2.putText(
                annotated,
                f"Confidence: {confidence*100:.2f}%",
                (panel_x1 + 10, panel_y1 + 49),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            cv2.putText(
                annotated,
                f"Empty: {empty_probability*100:.1f}%  "
                f"Full: {full_probability*100:.1f}%",
                (panel_x1 + 10, panel_y1 + 74),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

            # Frame number.
            cv2.putText(
                annotated,
                f"Frame: {original_frame_number}",
                (20, FRAME_HEIGHT - 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            # Original video timestamp.
            cv2.putText(
                annotated,
                f"Time: {str(timedelta(seconds=int(frame_timestamp)))}",
                (20, FRAME_HEIGHT - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            writer.write(
                annotated
            )

            # ----------------------------------------------------------
            # SAVE PER-FRAME PREDICTION
            # ----------------------------------------------------------

            results.append({

                "frame_number": original_frame_number,

                "timestamp_seconds": round(
                    frame_timestamp, 4
                ),

                "timestamp": str(
                    timedelta(seconds=frame_timestamp)
                ),

                "predicted_class_id": predicted_class_id,

                "predicted_class": predicted_class,

                "display_label": display_label,

                "confidence": round(
                    confidence, 6
                ),

                "empty_probability": round(
                    empty_probability, 6
                ),

                "full_probability": round(
                    full_probability, 6
                ),

                "inference_time_ms": round(
                    inference_time * 1000, 3
                )

            })

            processed_frames += 1

            progress.update(1)

            # Progress log every 500 frames.
            if processed_frames % 500 == 0:

                elapsed = (
                    time.time() - inference_start_time
                )

                processing_fps = (
                    processed_frames / elapsed
                    if elapsed > 0 else 0
                )

                print(
                    f"\nProcessed: {processed_frames}/{FRAMES_TO_PROCESS}"
                    f" | FULL: {prediction_counts['full']}"
                    f" | EMPTY: {prediction_counts['empty']}"
                    f" | UNCERTAIN: {prediction_counts['uncertain']}"
                    f" | FPS: {processing_fps:.2f}"
                )

finally:

    progress.close()

    cap.release()

    writer.release()


# ========================= 14. SAVE CSV ===============================

df = pd.DataFrame(results)

df.to_csv(
    OUTPUT_CSV_PATH,
    index=False
)


# ========================= 15. SUMMARY JSON ===========================

total_elapsed = (
    time.time() - inference_start_time
)

average_processing_fps = (
    processed_frames / total_elapsed
    if total_elapsed > 0 else 0
)

average_model_inference_ms = (
    (total_inference_time / processed_frames) * 1000
    if processed_frames > 0 else 0
)

summary = {

    "video_path": VIDEO_PATH,

    "model_path": MODEL_PATH,

    "output_video_path": OUTPUT_VIDEO_PATH,

    "output_csv_path": OUTPUT_CSV_PATH,

    "training_resolution": {
        "width": TRAIN_WIDTH,
        "height": TRAIN_HEIGHT
    },

    "inference_resolution": {
        "width": FRAME_WIDTH,
        "height": FRAME_HEIGHT
    },

    "roi_scaling": {
        "scale_x": SCALE_X,
        "scale_y": SCALE_Y
    },

    "original_training_roi": {
        "x1": TRAIN_X1,
        "y1": TRAIN_Y1,
        "x2": TRAIN_X2,
        "y2": TRAIN_Y2
    },

    "scaled_inference_roi": {
        "x1": X1,
        "y1": Y1,
        "x2": X2,
        "y2": Y2,
        "width": X2 - X1,
        "height": Y2 - Y1
    },

    "video": {
        "fps": VIDEO_FPS,
        "total_frames": TOTAL_FRAMES,
        "duration_seconds": VIDEO_DURATION
    },

    "selected_range": {
        "start_seconds": START_TIME,
        "end_seconds": END_TIME,
        "start_frame": START_FRAME,
        "end_frame_exclusive": END_FRAME
    },

    "class_mapping": {
        "empty": 0,
        "full": 1
    },

    "uncertainty_threshold": UNCERTAINTY_THRESHOLD,

    "processed_frames": processed_frames,

    "prediction_counts": prediction_counts,

    "average_processing_fps": average_processing_fps,

    "average_model_inference_ms": average_model_inference_ms,

    "total_processing_seconds": total_elapsed

}

with open(
    OUTPUT_JSON_PATH,
    "w"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )


# ========================= 16. FINAL REPORT ===========================

print("\n\n" + "=" * 75)
print("INFERENCE COMPLETED")
print("=" * 75)

print("Frames processed:", processed_frames)

print("FULL predictions:",
      prediction_counts["full"])

print("EMPTY predictions:",
      prediction_counts["empty"])

print("UNCERTAIN predictions:",
      prediction_counts["uncertain"])

print(
    f"\nTotal processing time: {total_elapsed:.2f} seconds"
)

print(
    f"Average processing FPS: {average_processing_fps:.2f}"
)

print(
    f"Average model inference time: "
    f"{average_model_inference_ms:.2f} ms/frame"
)

print("\nOUTPUT FILES")

print("\nAnnotated video:")
print(OUTPUT_VIDEO_PATH)

print("\nPrediction CSV:")
print(OUTPUT_CSV_PATH)

print("\nSummary JSON:")
print(OUTPUT_JSON_PATH)

print("\n" + "=" * 75)


# ========================= 17. DISPLAY VIDEO ==========================

try:

    from IPython.display import display, Video

    if os.path.isfile(OUTPUT_VIDEO_PATH):

        print("\nAnnotated video preview:")

        display(
            Video(
                OUTPUT_VIDEO_PATH,
                embed=False,
                width=1000
            )
        )

except Exception as e:

    print(
        "Could not display video preview:",
        e
    )

print("\nALL TASKS COMPLETED.")
