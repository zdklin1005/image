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


DETECTION_SUPPORTED_FRUITS = {
    "apple",
    "banana",
    "grape",
    "mango",
    "melon",
    "orange",
    "peach",
    "pear",
    "pineapple",
    "watermelon",
}

RIPENESS_SUPPORTED_FRUITS = {
    "apple",
    "banana",
    "grape",
    "mango",
    "melon",
    "orange",
    "peach",
    "pear",
}

DEFECT_SUPPORTED_FRUITS = set(
    RIPENESS_SUPPORTED_FRUITS
)

MODEL_A_EXTENSION_FRUITS = {
    "pineapple",
    "watermelon",
}

# Model A has no Peach or Mango class.  When the two detectors that do
# contain those classes agree on one of them, an unsupported Model A class
# (for example Apple or Orange) should not override that evidence.
MODEL_C_D_PRIORITY_FRUITS = {
    "mango",
    "peach",
}

MODEL_C_D_PRIORITY_MINIMUM_CONFIDENCE = 0.70

# Backwards-compatible name for code that only checks whether a fruit can be
# detected. Analysis capabilities are exposed separately below.
SUPPORTED_FRUITS = DETECTION_SUPPORTED_FRUITS

CLASS_CONFIDENCE_THRESHOLDS = {
    "apple": 0.30,
    "banana": 0.30,
    "grape": 0.35,
    "mango": 0.60,
    "melon": 0.35,
    "orange": 0.30,
    "peach": 0.35,
    "pear": 0.35,
}

DEFAULT_CLASS_CONFIDENCE_THRESHOLD = 0.50


def get_class_confidence_threshold(fruit_type):
    fruit_class = normalise_fruit_name(fruit_type)
    return CLASS_CONFIDENCE_THRESHOLDS.get(
        fruit_class,
        DEFAULT_CLASS_CONFIDENCE_THRESHOLD,
    )


def filter_detections_by_class_threshold(
    detections,
    confidence_floor=0.0,
):
    return [
        detection
        for detection in detections
        if detection["confidence"] >= max(
            confidence_floor,
            get_class_confidence_threshold(detection["fruit_type"]),
        )
    ]


# ============================================================
# MODEL A - FRUIT DETECTION
# ============================================================

def detect_with_model_a(
    image,
    confidence_threshold=0.40,
    apply_class_thresholds=True,
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

        if apply_class_thresholds and confidence < max(
            confidence_threshold,
            get_class_confidence_threshold(fruit_type),
        ):
            continue

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
    confidence_threshold=0.40,
    apply_class_thresholds=True,
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

        if apply_class_thresholds and confidence < max(
            confidence_threshold,
            get_class_confidence_threshold(fruit_type),
        ):
            continue

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
    confidence_threshold=0.30,
    apply_class_thresholds=True,
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

        if apply_class_thresholds and confidence < max(
            confidence_threshold,
            get_class_confidence_threshold(fruit_type),
        ):
            continue

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


# ============================================================
# FUSE MODEL A + MODEL C + MODEL D
# ============================================================

def fuse_detections(
    detections_a,
    detections_c,
    detections_d,
    iou_threshold=0.30,
    cross_class_iou_threshold=0.60
):
    """
    Fuse fruit detections from Models A, C and D.

    Same-class boxes retain the original IoU rule. Boxes with different
    predicted classes are also grouped when they overlap almost completely,
    because they are very likely to describe the same physical fruit. The
    final class is selected by confidence-weighted voting within each spatial
    group, while spatially separate fruits remain separate detections.
    """
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be between 0 and 1.")

    if not 0.0 <= cross_class_iou_threshold <= 1.0:
        raise ValueError(
            "cross_class_iou_threshold must be between 0 and 1."
        )

    # A stricter threshold is required for cross-class grouping so two
    # neighbouring fruits are not collapsed merely because their boxes touch.
    cross_class_iou_threshold = max(
        cross_class_iou_threshold,
        iou_threshold,
    )

    all_detections = sorted(
        detections_a + detections_c + detections_d,
        key=lambda detection: detection["confidence"],
        reverse=True,
    )

    spatial_groups = []

    for detection in all_detections:
        detection_class = normalise_fruit_name(
            detection["fruit_type"]
        )
        best_group = None
        best_group_iou = 0.0

        for group in spatial_groups:
            group_iou = 0.0

            for member in group:
                member_class = normalise_fruit_name(
                    member["fruit_type"]
                )
                current_iou = calculate_iou(
                    detection["bounding_box"],
                    member["bounding_box"],
                )
                required_iou = (
                    iou_threshold
                    if detection_class == member_class
                    else cross_class_iou_threshold
                )

                if current_iou >= required_iou:
                    group_iou = max(group_iou, current_iou)

            if group_iou > best_group_iou:
                best_group_iou = group_iou
                best_group = group

        if best_group is None:
            spatial_groups.append([detection])
        else:
            best_group.append(detection)

    final_detections = []

    for group in spatial_groups:
        # One vote per model prevents repeated predictions from a single model
        # from overpowering the other models in the ensemble.
        best_by_model = {}

        for detection in group:
            model_name = detection["model"]
            previous = best_by_model.get(model_name)

            if (
                previous is None
                or detection["confidence"] > previous["confidence"]
            ):
                best_by_model[model_name] = detection

        class_votes = {}

        for detection in best_by_model.values():
            fruit_class = normalise_fruit_name(
                detection["fruit_type"]
            )
            class_votes[fruit_class] = (
                class_votes.get(fruit_class, 0.0)
                + detection["confidence"]
            )

        model_a_detection = best_by_model.get("A")
        model_a_class = (
            normalise_fruit_name(
                model_a_detection["fruit_type"]
            )
            if model_a_detection is not None
            else None
        )

        model_c_detection = best_by_model.get("C")
        model_d_detection = best_by_model.get("D")
        model_c_class = (
            normalise_fruit_name(model_c_detection["fruit_type"])
            if model_c_detection is not None else None
        )
        model_d_class = (
            normalise_fruit_name(model_d_detection["fruit_type"])
            if model_d_detection is not None else None
        )

        # Targeted configuration-only correction for Peach/Mango.  This is
        # deliberately limited to a C+D agreement and does not alter normal
        # voting for the other fruit classes.
        model_c_d_priority_class = (
            model_c_class
            if (
                model_c_class == model_d_class
                and model_c_class in MODEL_C_D_PRIORITY_FRUITS
                and model_c_detection["confidence"]
                    + model_d_detection["confidence"]
                    >= MODEL_C_D_PRIORITY_MINIMUM_CONFIDENCE
            )
            else None
        )

        # Model A is the only detector trained for pineapple and watermelon.
        # When it identifies either extension fruit in a spatial group, retain
        # that class instead of allowing Models C/D (which do not contain those
        # classes) to relabel it as a visually similar core fruit.
        if model_a_class in MODEL_A_EXTENSION_FRUITS:
            winning_class = model_a_class
        elif model_c_d_priority_class is not None:
            winning_class = model_c_d_priority_class
        else:
            winning_class = max(
                class_votes,
                key=lambda fruit_class: (
                    class_votes[fruit_class],
                    max(
                        detection["confidence"]
                        for detection in best_by_model.values()
                        if normalise_fruit_name(
                            detection["fruit_type"]
                        ) == fruit_class
                    ),
                ),
            )

        winning_detections = [
            detection
            for detection in best_by_model.values()
            if normalise_fruit_name(
                detection["fruit_type"]
            ) == winning_class
        ]
        representative = max(
            winning_detections,
            key=lambda detection: detection["confidence"],
        )

        model_order = [
            model_name
            for model_name in ["A", "C", "D"]
            if model_name in best_by_model
        ]
        detected_classes = {
            normalise_fruit_name(detection["fruit_type"])
            for detection in best_by_model.values()
        }
        class_disagreement = len(detected_classes) > 1

        if len(model_order) == 1:
            agreement = f"Model {model_order[0]} only"
        elif class_disagreement:
            agreement = (
                "Spatial match: Model "
                + " + Model ".join(model_order)
                + f"; class vote: {winning_class.title()}"
            )
        else:
            agreement = "Model " + " + Model ".join(model_order)

        pairwise_ious = [
            calculate_iou(
                group[first_index]["bounding_box"],
                group[second_index]["bounding_box"],
            )
            for first_index in range(len(group))
            for second_index in range(first_index + 1, len(group))
        ]

        model_c_fruit_class = (
            normalise_fruit_name(model_c_detection["fruit_type"])
            if model_c_detection is not None else None
        )
        model_c_matches_final = (
            model_c_fruit_class == winning_class
            if model_c_fruit_class is not None else False
        )
        is_supported = winning_class in SUPPORTED_FRUITS
        displayed_fruit_type = (
            winning_class.title()
            if is_supported else "Unsupported"
        )

        final_detections.append({
            "model": representative["model"],
            "fruit_type": displayed_fruit_type,
            "detected_fruit_type": winning_class.title(),
            "is_supported": is_supported,
            "confidence": representative["confidence"],
            "bounding_box": representative["bounding_box"],
            "models": model_order,
            "confidence_a": (
                best_by_model["A"]["confidence"]
                if "A" in best_by_model else None
            ),
            "confidence_c": (
                best_by_model["C"]["confidence"]
                if "C" in best_by_model else None
            ),
            "confidence_d": (
                best_by_model["D"]["confidence"]
                if "D" in best_by_model else None
            ),
            "model_c_ripeness": (
                model_c_detection.get("ripeness")
                if (
                    model_c_detection is not None
                    and model_c_matches_final
                    and is_supported
                ) else None
            ),
            "model_c_raw_ripeness": (
                model_c_detection.get("ripeness")
                if model_c_detection is not None else None
            ),
            "model_c_fruit_type": (
                model_c_fruit_class.title()
                if model_c_fruit_class is not None else None
            ),
            "model_c_matches_final": model_c_matches_final,
            "iou": max(pairwise_ious, default=0.0),
            "agreement": agreement,
            "class_disagreement": class_disagreement,
            "class_votes": {
                fruit_class.title(): vote
                for fruit_class, vote in class_votes.items()
            },
        })

    final_detections.sort(
        key=lambda detection: detection["confidence"],
        reverse=True,
    )

    return final_detections


# ============================================================
# DETECTION RELIABILITY AND BOUNDING-BOX VALIDATION
# ============================================================

def assess_detection_quality(
    detections,
    image_shape,
    valid_content_bbox=None,
    minimum_area_ratio=0.0025,
    minimum_content_overlap=0.25,
    maximum_aspect_ratio=12.0,
    retain_rejected=False,
):
    """Validate boxes and attach an operator-facing reliability status."""
    image_height, image_width = image_shape[:2]
    image_area = max(1, image_width * image_height)

    if valid_content_bbox is None:
        valid_content_bbox = (0, 0, image_width, image_height)

    content_x1, content_y1, content_x2, content_y2 = valid_content_bbox
    assessed = []

    for detection in detections:
        result = dict(detection)
        x1, y1, x2, y2 = result["bounding_box"]
        box_width = max(0, x2 - x1)
        box_height = max(0, y2 - y1)
        box_area = box_width * box_height
        area_ratio = box_area / image_area
        smaller_side = max(1, min(box_width, box_height))
        aspect_ratio = max(box_width, box_height) / smaller_side

        intersection_width = max(
            0,
            min(x2, content_x2) - max(x1, content_x1),
        )
        intersection_height = max(
            0,
            min(y2, content_y2) - max(y1, content_y1),
        )
        content_overlap = (
            intersection_width * intersection_height / box_area
            if box_area > 0 else 0.0
        )
        box_issues = []
        box_review_reasons = []

        if box_width < 8 or box_height < 8 or box_area <= 0:
            box_issues.append("Malformed or extremely small box")

        if area_ratio < minimum_area_ratio:
            box_issues.append("Box is too small for reliable analysis")

        if content_overlap < minimum_content_overlap:
            box_issues.append("Box overlaps image padding or lies outside content")
        elif content_overlap < 0.80:
            box_review_reasons.append(
                "Box partly overlaps letterbox padding"
            )

        if aspect_ratio > maximum_aspect_ratio:
            box_issues.append("Box has an implausible aspect ratio")

        if box_issues:
            box_status = "Rejected"
        elif box_review_reasons:
            box_status = "Review required"
        else:
            box_status = "Accepted"
        models = result.get("models", [])
        reliability_reasons = []

        if box_status == "Rejected":
            reliability_status = "Rejected"
            reliability_reasons.extend(box_issues)
        elif not result.get("is_supported", True):
            reliability_status = "Rejected"
            reliability_reasons.append("Fruit class is unsupported")
        elif box_status == "Review required":
            reliability_status = "Review required"
            reliability_reasons.extend(box_review_reasons)
        elif result.get("class_disagreement", False):
            reliability_status = "Review required"
            reliability_reasons.append("Detection models disagree on fruit class")
        elif len(models) < 2:
            reliability_status = "Review required"
            reliability_reasons.append("Detection is supported by one model only")
        else:
            reliability_status = "Accepted"
            reliability_reasons.append("Multiple models support the detection")

        result.update({
            "box_status": box_status,
            "box_area_ratio": float(area_ratio),
            "box_content_overlap": float(content_overlap),
            "box_aspect_ratio": float(aspect_ratio),
            "box_issues": box_issues,
            "box_review_reasons": box_review_reasons,
            "reliability_status": reliability_status,
            "reliability_reasons": reliability_reasons,
            "class_confidence_threshold": get_class_confidence_threshold(
                result.get("detected_fruit_type", result["fruit_type"])
            ),
        })

        if retain_rejected or reliability_status != "Rejected":
            assessed.append(result)

    assessed.sort(
        key=lambda detection: detection["confidence"],
        reverse=True,
    )
    return assessed


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

        reliability_status = detection.get(
            "reliability_status",
            "Review required",
        )
        colour = {
            "Accepted": (0, 180, 0),
            "Review required": (0, 165, 255),
            "Rejected": (128, 128, 128),
        }.get(reliability_status, (0, 0, 255))

        cv2.rectangle(
            output_image,
            (x1, y1),
            (x2, y2),
            colour,
            3
        )

        label = (
            f"{fruit_type} "
            f"{confidence * 100:.1f}% "
            f"[{reliability_status}]"
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
            colour,
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
