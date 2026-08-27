"""Bridge between the Streamlit UI and the teammates' pipeline."""
import os
import pathlib
import tempfile

import cv2

from preprocessing import preprocess_fruit_image
from calibration_segmentation.segmentation import segment_fruit_otsu, refine_fruit_mask
from calibration_segmentation.measurement import extract_main_fruit


def to_rgb(img):
    """OpenCV is BGR, Streamlit expects RGB. Single-channel masks pass through."""
    if img is None or img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def preprocess_and_segment(file_bytes: bytes, suffix: str = ".jpg") -> dict:
    """Stages 1-2: everything that doesn't depend on the confidence slider."""
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