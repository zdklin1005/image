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

MODEL_D_PATH = (
    PROJECT_ROOT
    / "models"
    / "model_d_best.pt"
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

model_d = YOLO(
    str(MODEL_D_PATH)
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
# MODEL D - FRUIT DETECTION
# ============================================================

def detect_with_model_d(
    image,
    confidence_threshold=0.30
):
    """
    Model D:
    Specialized fruit detector.

    Supported fruits:
        Apple
        Banana
        Grape
        Mango
        Melon
        Orange
        Peach
        Pear

    Returns:
        Fruit type
        Confidence
        Bounding box
    """

    results = model_d.predict(
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
            model_d.names[class_id]
        ).strip()

        x1, y1, x2, y2 = [
            int(value)
            for value
            in box.xyxy[0].cpu().tolist()
        ]

        detections.append({
            "model": "D",

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

def suppress_duplicate_detections(
    detections,
    iou_threshold=0.75
):
    """
    Remove highly overlapping detections that most
    likely represent the same physical fruit.

    Fruit class is ignored during this final check
    because different models may assign different
    classes to the same physical fruit.
    """

    if not detections:
        return []

    def detection_priority(detection):

        model_count = len(
            detection.get(
                "models",
                []
            )
        )

        confidence = detection.get(
            "confidence",
            0.0
        )

        return (
            model_count,
            confidence
        )

    sorted_detections = sorted(
        detections,
        key=detection_priority,
        reverse=True
    )

    kept = []

    for candidate in sorted_detections:

        is_duplicate = False

        for existing in kept:

            overlap = calculate_iou(
                candidate["bounding_box"],
                existing["bounding_box"]
            )

            if overlap >= iou_threshold:

                is_duplicate = True
                break

        if not is_duplicate:

            kept.append(
                candidate
            )

    return kept

# ============================================================
# FUSE MODEL A + MODEL C + MODEL D
# ============================================================

def fuse_detections(
    detections_a,
    detections_c,
    detections_d,
    iou_threshold=0.30
):
    """
    Fuse fruit detections from Models A, C and D.

    Detections are combined when:
        1. Fruit type is the same
        2. Bounding boxes overlap sufficiently

    Multiple fruits of the same type are still allowed
    when they are located at different positions.
    """

    # ========================================================
    # COMBINE ALL RAW DETECTIONS
    # ========================================================

    all_detections = (
        detections_a
        + detections_c
        + detections_d
    )

    # Highest confidence first
    all_detections = sorted(
        all_detections,
        key=lambda detection:
        detection["confidence"],
        reverse=True
    )

    final_detections = []

    # ========================================================
    # PROCESS EVERY DETECTION
    # ========================================================

    for detection in all_detections:

        fruit_type = normalise_fruit_name(
            detection["fruit_type"]
        )

        detection_box = detection[
            "bounding_box"
        ]

        matching_detection = None
        best_iou = 0.0

        # ====================================================
        # LOOK FOR SAME PHYSICAL FRUIT
        # ====================================================

        for final_detection in final_detections:

            final_fruit_type = (
                normalise_fruit_name(
                    final_detection[
                        "fruit_type"
                    ]
                )
            )

            # Different fruit type
            if fruit_type != final_fruit_type:
                continue

            current_iou = calculate_iou(
                detection_box,
                final_detection[
                    "bounding_box"
                ]
            )

            if (
                current_iou >= iou_threshold
                and current_iou > best_iou
            ):

                best_iou = current_iou

                matching_detection = (
                    final_detection
                )

        # ====================================================
        # MATCH FOUND
        # ====================================================

        if matching_detection is not None:

            current_model = detection[
                "model"
            ]

            # Add model if not already included
            if (
                current_model
                not in matching_detection["models"]
            ):

                matching_detection[
                    "models"
                ].append(
                    current_model
                )

            # -----------------------------------------------
            # SAVE MODEL-SPECIFIC CONFIDENCE
            # -----------------------------------------------

            if current_model == "A":

                matching_detection[
                    "confidence_a"
                ] = detection[
                    "confidence"
                ]

            elif current_model == "C":

                matching_detection[
                    "confidence_c"
                ] = detection[
                    "confidence"
                ]

                # Save Model C ripeness
                matching_detection[
                    "model_c_ripeness"
                ] = detection.get(
                    "ripeness"
                )

            elif current_model == "D":

                matching_detection[
                    "confidence_d"
                ] = detection[
                    "confidence"
                ]

            # -----------------------------------------------
            # USE HIGHER-CONFIDENCE BOUNDING BOX
            # -----------------------------------------------

            if (
                detection["confidence"]
                >
                matching_detection["confidence"]
            ):

                matching_detection[
                    "confidence"
                ] = detection[
                    "confidence"
                ]

                matching_detection[
                    "bounding_box"
                ] = detection[
                    "bounding_box"
                ]

                matching_detection[
                    "model"
                ] = detection[
                    "model"
                ]

            matching_detection[
                "iou"
            ] = max(
                matching_detection.get(
                    "iou",
                    0.0
                ),
                best_iou
            )

        # ====================================================
        # NEW PHYSICAL FRUIT
        # ====================================================

        else:

            new_detection = {
                "model": detection["model"],

                "fruit_type": (
                    fruit_type.title()
                ),

                "confidence": detection[
                    "confidence"
                ],

                "bounding_box": detection[
                    "bounding_box"
                ],

                "models": [
                    detection["model"]
                ],

                "confidence_a": None,
                "confidence_c": None,
                "confidence_d": None,

                "model_c_ripeness": None,

                "iou": 0.0
            }

            # -----------------------------------------------
            # MODEL-SPECIFIC INFORMATION
            # -----------------------------------------------

            if detection["model"] == "A":

                new_detection[
                    "confidence_a"
                ] = detection[
                    "confidence"
                ]

            elif detection["model"] == "C":

                new_detection[
                    "confidence_c"
                ] = detection[
                    "confidence"
                ]

                new_detection[
                    "model_c_ripeness"
                ] = detection.get(
                    "ripeness"
                )

            elif detection["model"] == "D":

                new_detection[
                    "confidence_d"
                ] = detection[
                    "confidence"
                ]

            final_detections.append(
                new_detection
            )

    # ========================================================
    # CREATE AGREEMENT TEXT
    # ========================================================

    for detection in final_detections:

        models = detection[
            "models"
        ]

        model_order = [
            model
            for model in ["A", "C", "D"]
            if model in models
        ]

        if len(model_order) == 1:

            detection[
                "agreement"
            ] = (
                f"Model {model_order[0]} only"
            )

        else:

            detection[
                "agreement"
            ] = (
                "Model "
                + " + Model ".join(
                    model_order
                )
            )


    # ========================================================
    # FINAL CROSS-CLASS DUPLICATE SUPPRESSION
    # ========================================================

    final_detections = suppress_duplicate_detections(
        final_detections,
        iou_threshold=0.75
    )


    # ========================================================
    # SORT FINAL FRUITS BY CONFIDENCE
    # ========================================================

    final_detections.sort(
        key=lambda detection:
        detection["confidence"],
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
