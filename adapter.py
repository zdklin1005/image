"""Bridge between the Streamlit UI and the teammates' pipeline."""
import os
import pathlib
import tempfile

import cv2

from preprocessing import preprocess_fruit_image
from calibration_segmentation.segmentation import segment_fruit_otsu, refine_fruit_mask
from calibration_segmentation.measurement import extract_main_fruit
from fruit_ripeness_object_detection.blemish import detect_fruit_blemish


def to_rgb(img):
    if img is None or img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def preprocess_and_segment(file_bytes: bytes, suffix: str = ".jpg") -> dict:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(file_bytes)
        tmp.close()
        pre = preprocess_fruit_image(tmp.name)
    finally:
        os.unlink(tmp.name)

    working = pre["analysis_image"].copy()

    (gray_image, gray_mask, gray_threshold,
     sat_image, sat_mask, sat_threshold) = segment_fruit_otsu(working)

    opened_mask, refined_mask = refine_fruit_mask(
        sat_mask, opening_kernel_size=3, closing_kernel_size=5
    )
    fruit_mask, fruit_contour, fruit_area_pixels = extract_main_fruit(refined_mask)

    return {
        "working_image": working,
        "display_image": pre["display_image"],
        "blur_score": pre["blur_score"],
        "is_blurry": pre["is_blurry"],
        "gray_mask": gray_mask,
        "gray_threshold": gray_threshold,
        "saturation_mask": sat_mask,
        "saturation_threshold": sat_threshold,
        "refined_mask": refined_mask,
        "fruit_mask": fruit_mask,
        "fruit_contour": fruit_contour,
        "fruit_area_pixels": fruit_area_pixels,
    }


def compute_fruit_detail(working_image, bounding_box, fruit_type):
    """
    Per-fruit area + blemish analysis. Crops to one YOLO detection's
    bounding box, re-segments just that crop, then runs fruit-specific
    blemish detection on it.

    Returns a dict — blemish fields are None if segmentation on the
    crop failed, so the UI can show "N/A" instead of crashing.
    """
    empty = {
        "area_pixels": None, "mask_status": None,
        "blemish_area_pixels": None, "blemish_percentage": None, "blemish_overlay": None,
    }

    try:
        x1, y1, x2, y2 = [int(v) for v in bounding_box]
    except (TypeError, ValueError):
        return {**empty, "mask_status": "Invalid bounding box"}

    crop = working_image[max(y1, 0):y2, max(x1, 0):x2]
    if crop.size == 0:
        return {**empty, "mask_status": "Invalid crop"}

    try:
        _, _, _, _, sat_mask, _ = segment_fruit_otsu(crop)
        _, refined_mask = refine_fruit_mask(sat_mask, opening_kernel_size=3, closing_kernel_size=5)
        fruit_mask, _, area_pixels = extract_main_fruit(refined_mask)
        if area_pixels is None or area_pixels <= 0:
            return {**empty, "mask_status": "Segmentation failed"}
    except Exception:
        return {**empty, "mask_status": "Segmentation failed"}

    try:
        blemish = detect_fruit_blemish(crop, fruit_mask, fruit_type)
        return {
            "area_pixels": area_pixels,
            "mask_status": "Valid",
            "blemish_area_pixels": blemish["blemish_area_pixels"],
            "blemish_percentage": blemish["blemish_percentage"],
            "blemish_overlay": blemish["blemish_overlay"],
        }
    except Exception as e:
        import traceback
        print("BLEMISH DETECTION FAILED:")
        traceback.print_exc()
        return {
            "area_pixels": area_pixels, "mask_status": "Valid",
            "blemish_area_pixels": None, "blemish_percentage": None, "blemish_overlay": None,
        }