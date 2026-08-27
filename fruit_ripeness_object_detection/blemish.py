from pathlib import Path

import cv2
import numpy as np


# ============================================================
# FRUIT-SPECIFIC HSV THRESHOLDS
# ============================================================

def get_fruit_specific_hsv_ranges(fruit_type):
    """
    Return HSV ranges for blemish detection
    based on fruit type.

    Each fruit has a list of HSV ranges:
    [
        (lower1, upper1),
        (lower2, upper2),
        ...
    ]
    """

    fruit_type = str(fruit_type).strip().lower()

    # Default ranges if fruit not found
    default_ranges = [
        (
            np.array([0, 0, 0]),      # very dark / black
            np.array([179, 255, 65])
        ),
        (
            np.array([5, 40, 0]),     # dark brown
            np.array([25, 255, 110])
        )
    ]

    fruit_ranges = {
        "apple": [
            (
                np.array([0, 0, 0]),
                np.array([179, 255, 60])
            ),
            (
                np.array([5, 40, 0]),
                np.array([20, 255, 95])
            )
        ],

        "banana": [
            (
                np.array([0, 0, 0]),
                np.array([179, 255, 70])
            ),
            (
                np.array([5, 45, 0]),
                np.array([25, 255, 125])
            )
        ],

        "grape": [
            (
                np.array([0, 0, 0]),
                np.array([179, 255, 70])
            ),
            (
                np.array([5, 40, 0]),
                np.array([20, 255, 90])
            )
        ],

        "mango": [
            (
                np.array([0, 0, 0]),
                np.array([179, 255, 65])
            ),
            (
                np.array([5, 40, 0]),
                np.array([25, 255, 115])
            )
        ],

        "melon": [
            (
                np.array([0, 0, 0]),
                np.array([179, 255, 60])
            ),
            (
                np.array([5, 30, 0]),
                np.array([25, 220, 100])
            )
        ],

        "orange": [
            (
                np.array([0, 0, 0]),
                np.array([179, 255, 60])
            ),
            (
                np.array([5, 45, 0]),
                np.array([20, 255, 95])
            )
        ],

        "peach": [
            (
                np.array([0, 0, 0]),
                np.array([179, 255, 60])
            ),
            (
                np.array([5, 35, 0]),
                np.array([20, 255, 105])
            )
        ],

        "pear": [
            (
                np.array([0, 0, 0]),
                np.array([179, 255, 65])
            ),
            (
                np.array([5, 35, 0]),
                np.array([18, 255, 110])
            ),
            (
                np.array([35, 20, 20]),    # possible mold / green-blue patch
                np.array([100, 170, 170])
            )
        ]
    }

    return fruit_ranges.get(fruit_type, default_ranges)


# ============================================================
# HELPER: ENSURE BINARY MASK
# ============================================================

def ensure_binary_mask(mask):
    """
    Convert input mask to single-channel binary mask.
    """

    if mask is None:
        raise ValueError("Mask cannot be None.")

    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    _, binary_mask = cv2.threshold(
        mask,
        127,
        255,
        cv2.THRESH_BINARY
    )

    return binary_mask


# ============================================================
# HELPER: REMOVE SMALL NOISE
# ============================================================

def remove_small_components(binary_mask, min_area=60):
    """
    Remove very small white blobs from a binary mask.
    """

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask,
        connectivity=8
    )

    cleaned_mask = np.zeros_like(binary_mask)

    for label_index in range(1, num_labels):
        area = stats[label_index, cv2.CC_STAT_AREA]

        if area >= min_area:
            cleaned_mask[labels == label_index] = 255

    return cleaned_mask


# ============================================================
# HELPER: CREATE OVERLAY IMAGE
# ============================================================

def create_blemish_overlay(image, blemish_mask):
    """
    Overlay blemish regions in red on the original image.
    """

    if image is None:
        raise ValueError("Image cannot be None.")

    if blemish_mask is None:
        raise ValueError("Blemish mask cannot be None.")

    output = image.copy()

    red_layer = np.zeros_like(output)
    red_layer[:] = (0, 0, 255)

    mask_indices = blemish_mask > 0

    output[mask_indices] = cv2.addWeighted(
        output[mask_indices],
        0.35,
        red_layer[mask_indices],
        0.65,
        0
    )

    return output


# ============================================================
# MAIN: FRUIT-SPECIFIC BLEMISH ANALYSIS
# ============================================================

def detect_fruit_blemish(
    image,
    fruit_mask,
    fruit_type,
    opening_kernel_size=3,
    closing_kernel_size=5,
    min_component_area=60
):
    """
    Detect blemish regions inside a fruit using
    fruit-specific HSV thresholds.

    Parameters:
        image:
            OpenCV BGR image.

        fruit_mask:
            Binary fruit mask from segmentation.

        fruit_type:
            Fruit type string such as
            "Apple", "Banana", "Pear", etc.

        opening_kernel_size:
            Kernel size for morphological opening.

        closing_kernel_size:
            Kernel size for morphological closing.

        min_component_area:
            Minimum connected component size to keep.

    Returns:
        Dictionary containing blemish analysis results.
    """

    if image is None:
        raise ValueError("Image cannot be None.")

    if fruit_mask is None:
        raise ValueError("Fruit mask cannot be None.")

    fruit_mask_binary = ensure_binary_mask(fruit_mask)

    hsv_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    hsv_ranges = get_fruit_specific_hsv_ranges(fruit_type)

    combined_mask = np.zeros(
        fruit_mask_binary.shape,
        dtype=np.uint8
    )

    # Apply all HSV ranges for this fruit
    for lower, upper in hsv_ranges:
        current_mask = cv2.inRange(
            hsv_image,
            lower,
            upper
        )

        combined_mask = cv2.bitwise_or(
            combined_mask,
            current_mask
        )

    # Keep only pixels inside the fruit region
    blemish_mask = cv2.bitwise_and(
        combined_mask,
        fruit_mask_binary
    )

    # Morphological cleanup
    opening_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (opening_kernel_size, opening_kernel_size)
    )

    closing_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (closing_kernel_size, closing_kernel_size)
    )

    blemish_mask = cv2.morphologyEx(
        blemish_mask,
        cv2.MORPH_OPEN,
        opening_kernel
    )

    blemish_mask = cv2.morphologyEx(
        blemish_mask,
        cv2.MORPH_CLOSE,
        closing_kernel
    )

    # Remove tiny noise
    blemish_mask = remove_small_components(
        blemish_mask,
        min_area=min_component_area
    )

    fruit_pixels = cv2.countNonZero(
        fruit_mask_binary
    )

    blemish_pixels = cv2.countNonZero(
        blemish_mask
    )

    if fruit_pixels > 0:
        blemish_percentage = (
            blemish_pixels / fruit_pixels
        ) * 100
    else:
        blemish_percentage = 0.0

    overlay_image = create_blemish_overlay(
        image,
        blemish_mask
    )

    return {
        "fruit_type_used": fruit_type,
        "blemish_mask": blemish_mask,
        "blemish_overlay": overlay_image,
        "blemish_area_pixels": blemish_pixels,
        "fruit_area_pixels": fruit_pixels,
        "blemish_percentage": blemish_percentage
    }
