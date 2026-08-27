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

print("\n==============================")
print("CHECKING MODEL CLASSES")
print("==============================")

print("\nMODEL A PATH:")
print(MODEL_A_PATH)

print("\nMODEL A CLASSES:")
print(model_a.names)

print("\nMODEL C PATH:")
print(MODEL_C_PATH)

print("\nMODEL C CLASSES:")
print(model_c.names)

print("==============================\n")


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
# FUSE MULTIPLE DETECTIONS FROM MODEL A + MODEL C
# ============================================================

def fuse_detections(
    detections_a,
    detections_c,
    iou_threshold=0.30
):
    """
    Fuse multiple fruit detections from Model A and Model C.

    Rules:
    1. If A and C detect the same fruit at the same location,
       combine them into one detection.
    2. If only one model detects a fruit, keep it.
    3. Multiple different fruits can remain in the final result.

    Returns:
        List of final fruit detections.
    """

    final_detections = []

    # Keep track of Model C detections
    # that have already been matched.
    matched_c_indices = set()

    # ========================================================
    # PROCESS EACH MODEL A DETECTION
    # ========================================================

    for detection_a in detections_a:

        fruit_a = normalise_fruit_name(
            detection_a["fruit_type"]
        )

        best_match_index = None
        best_match_iou = 0.0

        # Try to find the same fruit from Model C
        for c_index, detection_c in enumerate(
            detections_c
        ):

            if c_index in matched_c_indices:
                continue

            fruit_c = normalise_fruit_name(
                detection_c["fruit_type"]
            )

            # Fruit type must agree
            if fruit_a != fruit_c:
                continue

            current_iou = calculate_iou(
                detection_a["bounding_box"],
                detection_c["bounding_box"]
            )

            if (
                current_iou >= iou_threshold
                and current_iou > best_match_iou
            ):
                best_match_iou = current_iou
                best_match_index = c_index

        # ====================================================
        # A + C AGREE
        # ====================================================

        if best_match_index is not None:

            detection_c = detections_c[
                best_match_index
            ]

            matched_c_indices.add(
                best_match_index
            )

            # Use bounding box from the
            # higher-confidence model
            if (
                detection_a["confidence"]
                >= detection_c["confidence"]
            ):

                selected = detection_a.copy()

            else:

                selected = detection_c.copy()

            selected["fruit_type"] = (
                fruit_a.title()
            )

            selected["agreement"] = (
                "Model A + Model C"
            )

            selected["models"] = [
                "A",
                "C"
            ]

            selected["iou"] = (
                best_match_iou
            )

            selected["confidence_a"] = (
                detection_a["confidence"]
            )

            selected["confidence_c"] = (
                detection_c["confidence"]
            )

            # Store Model C ripeness
            # for later ripeness fusion
            selected["model_c_ripeness"] = (
                detection_c.get(
                    "ripeness"
                )
            )

            final_detections.append(
                selected
            )

        # ====================================================
        # ONLY MODEL A DETECTED THIS FRUIT
        # ====================================================

        else:

            selected = detection_a.copy()

            selected["fruit_type"] = (
                fruit_a.title()
            )

            selected["agreement"] = (
                "Model A only"
            )

            selected["models"] = [
                "A"
            ]

            selected["confidence_a"] = (
                detection_a["confidence"]
            )

            selected["confidence_c"] = None

            selected["model_c_ripeness"] = None

            final_detections.append(
                selected
            )

    # ========================================================
    # ADD UNMATCHED MODEL C DETECTIONS
    # ========================================================

    for c_index, detection_c in enumerate(
        detections_c
    ):

        if c_index in matched_c_indices:
            continue

        fruit_c = normalise_fruit_name(
            detection_c["fruit_type"]
        )

        selected = detection_c.copy()

        selected["fruit_type"] = (
            fruit_c.title()
        )

        selected["agreement"] = (
            "Model C only"
        )

        selected["models"] = [
            "C"
        ]

        selected["confidence_a"] = None

        selected["confidence_c"] = (
            detection_c["confidence"]
        )

        selected["model_c_ripeness"] = (
            detection_c.get(
                "ripeness"
            )
        )

        final_detections.append(
            selected
        )

    # ========================================================
    # SORT BY CONFIDENCE
    # ========================================================

    final_detections.sort(
        key=lambda item:
        item["confidence"],
        reverse=True
    )

    return final_detections


# ============================================================
# DRAW ALL FINAL DETECTIONS
# ============================================================

def draw_final_detections(
    image,
    final_detections
):
    """
    Draw all final fruit bounding boxes.
    """

    output_image = image.copy()

    for detection in final_detections:

        fruit_type = detection[
            "fruit_type"
        ]

        confidence = detection[
            "confidence"
        ]

        x1, y1, x2, y2 = (
            detection[
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
            (
                x1,
                max(y1 - 10, 25)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            3
        )

    return output_image


# ============================================================
# CROP ONE DETECTED FRUIT
# ============================================================

def crop_detected_fruit(
    image,
    bounding_box,
    margin_ratio=0.10
):
    """
    Crop one detected fruit with a small margin.
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

    return image[
        y1:y2,
        x1:x2
    ].copy()


# ============================================================
# CROP ALL DETECTED FRUITS
# ============================================================

def crop_all_detected_fruits(
    image,
    final_detections,
    margin_ratio=0.10
):
    """
    Crop every detected fruit.

    Returns a list containing fruit type,
    bounding box and cropped image.
    """

    fruit_crops = []

    for index, detection in enumerate(
        final_detections,
        start=1
    ):

        crop = crop_detected_fruit(
            image,
            detection["bounding_box"],
            margin_ratio=margin_ratio
        )

        if crop.size == 0:
            continue

        fruit_crops.append({
            "index": index,
            "fruit_type": detection[
                "fruit_type"
            ],
            "bounding_box": detection[
                "bounding_box"
            ],
            "crop": crop
        })

    return fruit_crops
