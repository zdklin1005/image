"""
pipeline.py - Orchestrates the full fruit analysis pipeline across your
teammates' three modules:

1. DETECTION (fruit_detection.py): three YOLO models (A, C, D) each
   detect fruits independently. fuse_detections() merges their boxes
   into one list of physical fruits (matching by fruit type + IoU
   overlap), carrying Model C's rough ripeness guess along for later.

2. RIPENESS (ripeness_classification.py): each detected fruit is
   cropped out and classified by Model B and Model E, then fused
   together with Model C's guess via fuse_ripeness() (weighted:
   E 60%, C 25%, B 15%).

3. BLEMISH (blemish.py): each crop is segmented (reusing the existing
   calibration_segmentation package to isolate the fruit body), then
   scanned for defects using fruit-specific HSV thresholds.

analyze_image() ties all three stages together and returns one
unified result per detected fruit, plus an annotated image.
"""

import cv2

from fruit_ripeness_object_detection.fruit_detection import (
    detect_with_model_a,
    detect_with_model_c,
    detect_with_model_d,
    fuse_detections,
    crop_all_detected_fruits,
)
from fruit_ripeness_object_detection.ripeness_classification import (
    classify_with_model_b,
    classify_with_model_e,
    fuse_ripeness,
)
from fruit_ripeness_object_detection.blemish import detect_fruit_blemish

from calibration_segmentation.segmentation import segment_fruit_otsu, refine_fruit_mask
from calibration_segmentation.measurement import extract_main_fruit, calculate_projected_area_cm2


def _segment_fruit(crop_bgr):
    """
    Segments the dominant fruit body out of a cropped fruit image.
    Used for both blemish detection and size measurement, so it's
    only run once per fruit. Returns (fruit_mask, fruit_area_pixels)
    or (None, None) if segmentation fails (e.g. the crop is too
    small/uniform for a contour to be found).
    """
    try:
        (
            gray_image, gray_mask, gray_threshold,
            saturation_image, saturation_mask, saturation_threshold,
        ) = segment_fruit_otsu(crop_bgr)
        _, refined_mask = refine_fruit_mask(saturation_mask)
        fruit_mask, _, fruit_area_pixels = extract_main_fruit(refined_mask)
        return fruit_mask, fruit_area_pixels
    except ValueError:
        return None, None


def analyze_image(
    image_bgr,
    confidence_threshold_a=0.40,
    confidence_threshold_c=0.40,
    confidence_threshold_d=0.30,
    iou_threshold=0.30,
    pixels_per_cm=20.0,
):
    """
    Runs the full pipeline on one BGR image.

    pixels_per_cm: a fixed/preset spatial scale used to convert each
    fruit's projected area from pixels to cm^2. No perspective/manual
    calibration is applied here -- see sizing.py's manual mode if a
    per-photo calibrated scale is needed later.

    Returns:
        results: list of per-fruit dicts:
            fruit_type, detection_confidence, agreement, bounding_box,
            ripeness, ripeness_confidence, blemish_percentage,
            blemish_overlay (or None if segmentation failed),
            fruit_area_cm2 (or None if segmentation failed), crop
        annotated_image: BGR image with bounding boxes + labels drawn
    """
    detections_a = detect_with_model_a(image_bgr, confidence_threshold_a)
    detections_c = detect_with_model_c(image_bgr, confidence_threshold_c)
    detections_d = detect_with_model_d(image_bgr, confidence_threshold_d)

    final_detections = fuse_detections(
        detections_a, detections_c, detections_d, iou_threshold=iou_threshold
    )

    fruit_crops = crop_all_detected_fruits(image_bgr, final_detections)
    crops_by_index = {c["index"]: c["crop"] for c in fruit_crops}

    results = []

    for index, detection in enumerate(final_detections, start=1):
        crop = crops_by_index.get(index)
        if crop is None or crop.size == 0:
            continue

        fruit_type = detection["fruit_type"]

        # ---- Ripeness: fuse Model B + Model C + Model E ----
        result_b = classify_with_model_b(crop, fruit_type)
        result_e = classify_with_model_e(crop)
        ripeness_result = fuse_ripeness(
            result_b,
            detection.get("model_c_ripeness"),
            detection.get("confidence_c"),
            result_e,
        )

        # ---- Segmentation (shared by blemish detection + size) ----
        fruit_mask, fruit_area_pixels = _segment_fruit(crop)

        blemish_result = (
            detect_fruit_blemish(crop, fruit_mask, fruit_type)
            if fruit_mask is not None
            else None
        )

        fruit_area_cm2 = (
            calculate_projected_area_cm2(fruit_area_pixels, pixels_per_cm, pixels_per_cm)
            if fruit_area_pixels is not None
            else None
        )

        results.append({
            "fruit_type": fruit_type,
            "detection_confidence": detection["confidence"],
            "agreement": detection["agreement"],
            "bounding_box": detection["bounding_box"],
            "ripeness": ripeness_result["ripeness"],
            "ripeness_confidence": ripeness_result["confidence"],
            "blemish_percentage": (
                blemish_result["blemish_percentage"] if blemish_result else None
            ),
            "blemish_overlay": (
                blemish_result["blemish_overlay"] if blemish_result else None
            ),
            "fruit_area_cm2": fruit_area_cm2,
            "crop": crop,
        })

    annotated_image = _draw_results(image_bgr, results)

    return results, annotated_image


def _draw_results(image_bgr, results):
    """
    Draws bounding boxes with a fruit type + ripeness + confidence
    label (extends fruit_detection.py's draw_final_detections, which
    only labels fruit type + confidence).
    """
    output = image_bgr.copy()
    for r in results:
        x1, y1, x2, y2 = r["bounding_box"]
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 0, 255), 3)
        label = (
            f"{r['fruit_type']} | {r['ripeness']} | "
            f"{r['detection_confidence']*100:.0f}%"
        )
        cv2.putText(
            output, label, (x1, max(y1 - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
        )
    return output