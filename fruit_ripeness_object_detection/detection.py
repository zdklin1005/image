from pathlib import Path

import cv2
from ultralytics import YOLO


# ============================================================
# LOAD YOLO MODEL
# ============================================================

# Get the project root folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Model location
MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"

# Load trained YOLO model
model = YOLO(str(MODEL_PATH))


# ============================================================
# FRUIT DETECTION + RIPENESS CLASSIFICATION
# ============================================================

def detect_fruit_ripeness(
    image,
    confidence_threshold=0.40
):
    """
    Detect fruits and classify their ripeness using YOLO.

    Parameters:
        image:
            OpenCV BGR image.

        confidence_threshold:
            Minimum confidence required for a detection.

    Returns:
        A list of detected fruits.
    """

    # Run YOLO prediction
    results = model.predict(
        source=image,
        conf=confidence_threshold,
        verbose=False
    )

    result = results[0]

    detections = []

    # Process each detected object
    for box in result.boxes:

        # Get predicted class ID
        class_id = int(box.cls[0])

        # Get confidence score
        confidence = float(box.conf[0])

        # Get class name
        # Example:
        # "Banana Ripe"
        # "Apple Rotten"
        full_class = model.names[class_id]

        # Separate fruit type and ripeness
        fruit_type, ripeness = full_class.rsplit(
            " ",
            1
        )

        # Get bounding box coordinates
        x1, y1, x2, y2 = (
            box.xyxy[0]
            .cpu()
            .numpy()
            .astype(int)
        )

        # Store detection result
        detections.append({
            "fruit_type": fruit_type,
            "ripeness": ripeness,
            "confidence": confidence,
            "bounding_box": (
                x1,
                y1,
                x2,
                y2
            )
        })

    return detections


# ============================================================
# DRAW BOUNDING BOXES AND LABELS
# ============================================================

def draw_detections(
    image,
    detections
):
    """
    Draw YOLO bounding boxes and fruit/ripeness labels
    onto the image.

    Parameters:
        image:
            OpenCV BGR image.

        detections:
            Detection results returned by
            detect_fruit_ripeness().

    Returns:
        Image with bounding boxes and labels.
    """

    # Make a copy so the original image
    # is not changed
    output_image = image.copy()

    # Draw every detected fruit
    for detection in detections:

        fruit_type = detection["fruit_type"]
        ripeness = detection["ripeness"]
        confidence = detection["confidence"]

        x1, y1, x2, y2 = (
            detection["bounding_box"]
        )

        # Draw bounding box
        cv2.rectangle(
            output_image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # Create label
        label = (
            f"{fruit_type} "
            f"{ripeness} "
            f"{confidence * 100:.1f}%"
        )

        # Draw label above bounding box
        cv2.putText(
            output_image,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    return output_image
