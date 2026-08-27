from pathlib import Path

import cv2
from ultralytics import YOLO


# ============================================================
# PROJECT / MODEL PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_A_PATH = (
    PROJECT_ROOT
    / "models"
    / "model_a_best.pt"
)

MODEL_C_PATH = (
    PROJECT_ROOT
    / "models"
    / "model_c_best.pt"
)


# ============================================================
# LOAD MODELS
# ============================================================

model_a = YOLO(
    str(MODEL_A_PATH)
)

model_c = YOLO(
    str(MODEL_C_PATH)
)


# ============================================================
# MODEL A - FRUIT DETECTION
# ============================================================

def detect_with_model_a(
    image,
    confidence_threshold=0.40
):
    """
    Model A:
    Detect fruit type and bounding box.

    Supported fruits:
    Apple
    Banana
    Grape
    Orange
    Pineapple
    Watermelon
    """

    results = model_a.predict(
        source=image,
        conf=confidence_threshold,
        verbose=False
    )

    detections = []

    for box in results[0].boxes:

        class_id = int(
            box.cls[0]
        )

        confidence = float(
            box.conf[0]
        )

        fruit_type = str(
            model_a.names[class_id]
        ).strip()

        x1, y1, x2, y2 = [
            int(value)
            for value
            in box.xyxy[0].cpu().tolist()
        ]

        detections.append({
            "model": "A",
            "fruit_type": fruit_type,
            "confidence": confidence,
            "bounding_box": (
                x1,
                y1,
                x2,
                y2
            ),
            "ripeness": None
        })

    return detections


# ============================================================
# MODEL C - FRUIT DETECTION
# ============================================================

def detect_with_model_c(
    image,
    confidence_threshold=0.40
):
    """
    Model C originally predicts:
        Fruit + Ripeness

    Example:
        Banana Ripe
        Pear Rotten

    During the FIRST detection stage,
    only fruit type + bounding box are used.

    Ripeness is stored for later use.
    """

    results = model_c.predict(
        source=image,
        conf=confidence_threshold,
        verbose=False
    )

    detections = []

    for box in results[0].boxes:

        class_id = int(
            box.cls[0]
        )

        confidence = float(
            box.conf[0]
        )

        full_class = str(
            model_c.names[class_id]
        ).replace(
            "_",
            " "
        ).replace(
            "-",
            " "
        ).strip()

        words = full_class.split()

        if len(words) >= 2:

            ripeness = words[-1]

            fruit_type = " ".join(
                words[:-1]
            )

        else:

            fruit_type = full_class
            ripeness = "Unknown"

        x1, y1, x2, y2 = [
            int(value)
            for value
            in box.xyxy[0].cpu().tolist()
        ]

        detections.append({
            "model": "C",
            "fruit_type": fruit_type,
            "confidence": confidence,
            "bounding_box": (
                x1,
                y1,
                x2,
                y2
            ),

            # Save but DO NOT treat it as
            # final ripeness yet
            "ripeness": ripeness
        })

    return detections


# ============================================================
# NORMALISE FRUIT NAME
# ============================================================

def normalise_fruit_name(
    fruit_type
):
    """
    Make fruit names easier to compare.

    Example:
        Apples -> apple
        Apple  -> apple
        Grapes -> grape
    """

    fruit_name = str(
        fruit_type
    ).strip().lower()

    name_mapping = {
        "apples": "apple",
        "apple": "apple",

        "bananas": "banana",
        "banana": "banana",

        "grapes": "grape",
        "grape": "grape",

        "oranges": "orange",
        "orange": "orange",

        "mangoes": "mango",
        "mangos": "mango",
        "mango": "mango",

        "melons": "melon",
        "melon": "melon",

        "peaches": "peach",
        "peach": "peach",

        "pears": "pear",
        "pear": "pear",

        "pineapples": "pineapple",
        "pineapple": "pineapple",

        "watermelons": "watermelon",
        "watermelon": "watermelon"
    }

    return name_mapping.get(
        fruit_name,
        fruit_name
    )


# ============================================================
# CALCULATE IoU
# ============================================================

def calculate_iou(
    box1,
    box2
):
    """
    Calculate overlap between two bounding boxes.
    """

    x1_a, y1_a, x2_a, y2_a = box1
    x1_b, y1_b, x2_b, y2_b = box2

    intersection_x1 = max(
        x1_a,
        x1_b
    )

    intersection_y1 = max(
        y1_a,
        y1_b
    )

    intersection_x2 = min(
        x2_a,
        x2_b
    )

    intersection_y2 = min(
        y2_a,
        y2_b
    )

    intersection_width = max(
        0,
        intersection_x2
        - intersection_x1
    )

    intersection_height = max(
        0,
        intersection_y2
        - intersection_y1
    )

    intersection_area = (
        intersection_width
        * intersection_height
    )

    area_a = (
        max(0, x2_a - x1_a)
        * max(0, y2_a - y1_a)
    )

    area_b = (
        max(0, x2_b - x1_b)
        * max(0, y2_b - y1_b)
    )

    union_area = (
        area_a
        + area_b
        - intersection_area
    )

    if union_area <= 0:
        return 0.0

    return (
        intersection_area
        / union_area
    )


# ============================================================
# SELECT / FUSE MODEL A + MODEL C
# ============================================================

def select_detection(
    detections_a,
    detections_c,
    iou_threshold=0.30
):
    """
    Select one primary fruit detection.

    Current version assumes the image contains
    one main fruit.

    If A and C agree on fruit type and their boxes
    overlap, the higher-confidence box is used.

    If they disagree, the highest-confidence
    prediction is selected for now.

    Model D will later be added as another vote.
    """

    if (
        len(detections_a) == 0
        and len(detections_c) == 0
    ):
        return None

    # Get strongest prediction from each model
    best_a = None
    best_c = None

    if detections_a:

        best_a = max(
            detections_a,
            key=lambda item:
            item["confidence"]
        )

    if detections_c:

        best_c = max(
            detections_c,
            key=lambda item:
            item["confidence"]
        )

    # Only C detected
    if best_a is None:

        final_detection = (
            best_c.copy()
        )

        final_detection[
            "agreement"
        ] = "Model C only"

        return final_detection

    # Only A detected
    if best_c is None:

        final_detection = (
            best_a.copy()
        )

        final_detection[
            "agreement"
        ] = "Model A only"

        return final_detection

    # ========================================================
    # BOTH MODELS DETECTED SOMETHING
    # ========================================================

    fruit_a = normalise_fruit_name(
        best_a["fruit_type"]
    )

    fruit_c = normalise_fruit_name(
        best_c["fruit_type"]
    )

    iou = calculate_iou(
        best_a["bounding_box"],
        best_c["bounding_box"]
    )

    # Same fruit + overlapping location
    if (
        fruit_a == fruit_c
        and iou >= iou_threshold
    ):

        if (
            best_a["confidence"]
            >= best_c["confidence"]
        ):

            final_detection = (
                best_a.copy()
            )

        else:

            final_detection = (
                best_c.copy()
            )

        final_detection[
            "fruit_type"
        ] = fruit_a.title()

        final_detection[
            "agreement"
        ] = "Model A + Model C"

        final_detection[
            "iou"
        ] = iou

        return final_detection

    # ========================================================
    # MODELS DISAGREE
    # ========================================================

    if (
        best_a["confidence"]
        >= best_c["confidence"]
    ):

        final_detection = (
            best_a.copy()
        )

    else:

        final_detection = (
            best_c.copy()
        )

    final_detection[
        "agreement"
    ] = "Models disagree"

    final_detection[
        "iou"
    ] = iou

    return final_detection


# ============================================================
# DRAW FINAL FRUIT DETECTION
# ============================================================

def draw_final_detection(
    image,
    final_detection
):
    """
    Draw the selected final fruit bounding box.
    """

    output_image = image.copy()

    if final_detection is None:
        return output_image

    fruit_type = final_detection[
        "fruit_type"
    ]

    confidence = final_detection[
        "confidence"
    ]

    x1, y1, x2, y2 = (
        final_detection[
            "bounding_box"
        ]
    )

    # Red bounding box
    cv2.rectangle(
        output_image,
        (x1, y1),
        (x2, y2),
        (0, 0, 255),
        3
    )

    label = (
        f"{fruit_type} "
        f"{confidence * 100:.1f}%"
    )

    cv2.putText(
        output_image,
        label,
        (x1, max(y1 - 10, 25)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        3
    )

    return output_image


# ============================================================
# CROP FINAL FRUIT
# ============================================================

def crop_detected_fruit(
    image,
    bounding_box,
    margin_ratio=0.10
):
    """
    Crop detected fruit with a small margin.
    """

    image_height, image_width = (
        image.shape[:2]
    )

    x1, y1, x2, y2 = (
        bounding_box
    )

    box_width = x2 - x1
    box_height = y2 - y1

    margin_x = int(
        box_width * margin_ratio
    )

    margin_y = int(
        box_height * margin_ratio
    )

    x1 = max(
        0,
        x1 - margin_x
    )

    y1 = max(
        0,
        y1 - margin_y
    )

    x2 = min(
        image_width,
        x2 + margin_x
    )

    y2 = min(
        image_height,
        y2 + margin_y
    )

    crop = image[
        y1:y2,
        x1:x2
    ].copy()

    return crop
