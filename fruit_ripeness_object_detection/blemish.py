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

    fruit_hsv_ranges = {
    fruit_hsv_ranges = {
        "apple": [
            # dark brown / black bruise
            # dark brown / black bruise
            (
                np.array([0, 40, 0]),
                np.array([25, 255, 120])
                np.array([0, 40, 0]),
                np.array([25, 255, 120])
            ),

            # grey / mould

            # grey / mould
            (
                np.array([0, 0, 40]),
                np.array([179, 80, 210])
            ),

            # light grey / white mould
            (
                np.array([0, 0, 150]),
                np.array([179, 65, 245])
                np.array([0, 0, 40]),
                np.array([179, 80, 210])
            ),

            # light grey / white mould
            (
                np.array([0, 0, 150]),
                np.array([179, 65, 245])
            )
        ],

        "banana": [
            # dark bruise
            # dark bruise
            (
                np.array([0, 20, 0]),
                np.array([30, 255, 110])
                np.array([0, 20, 0]),
                np.array([30, 255, 110])
            ),

            # grey mould

            # grey mould
            (
                np.array([0, 0, 40]),
                np.array([179, 80, 210])
            ),

            # light grey mould
            (
                np.array([0, 0, 150]),
                np.array([179, 65, 245])
                np.array([0, 0, 40]),
                np.array([179, 80, 210])
            ),

            # light grey mould
            (
                np.array([0, 0, 150]),
                np.array([179, 65, 245])
            )
        ],

        "grape": [
            # dark rotten spots only
            (
                np.array([0, 20, 0]),
                np.array([25, 255, 90])
            ),

            # grey / white mould
            (
                np.array([0, 0, 40]),
                np.array([179, 80, 210])
            ),
            (
                np.array([0, 0, 150]),
                np.array([179, 65, 245])
            )
        ],

        "orange": [
            # black / dark brown
            # dark rotten spots only
            (
                np.array([0, 20, 0]),
                np.array([25, 255, 90])
            ),

            # grey / white mould
            (
                np.array([0, 0, 40]),
                np.array([179, 80, 210])
            ),
            (
                np.array([0, 0, 150]),
                np.array([179, 65, 245])
            )
        ],

        "orange": [
            # black / dark brown
            (
                np.array([0, 0, 0]),
                np.array([179, 255, 60])
                np.array([179, 255, 60])
            ),


            (
                np.array([5, 45, 0]),
                np.array([20, 255, 95])
            ),

            # green / blue-green mould
            (
                np.array([35, 15, 25]),
                np.array([115, 210, 220])
            ),

            # grey mould
            (
                np.array([0, 0, 40]),
                np.array([179, 80, 210])
            ),

            # light grey / white mould
            (
                np.array([0, 0, 150]),
                np.array([179, 65, 245])
                np.array([5, 45, 0]),
                np.array([20, 255, 95])
            ),

            # green / blue-green mould
            (
                np.array([35, 15, 25]),
                np.array([115, 210, 220])
            ),

            # grey mould
            (
                np.array([0, 0, 40]),
                np.array([179, 80, 210])
            ),

            # light grey / white mould
            (
                np.array([0, 0, 150]),
                np.array([179, 65, 245])
            )
        ],

        "mango": [
            # very dark / black rotten area
            # very dark / black rotten area
            (
                np.array([0, 0, 0]),
                np.array([179, 255, 75])
                np.array([179, 255, 75])
            ),

            # dark brown bruise / rot

            # dark brown bruise / rot
            (
                np.array([5, 40, 20]),
                np.array([5, 40, 20]),
                np.array([25, 255, 115])
            ),

            # grey mould
            (
                np.array([0, 0, 75]),
                np.array([179, 55, 170])
            ),

            # light grey / white mould
            (
                np.array([0, 0, 170]),
                np.array([179, 45, 235])
            ),

            # grey mould
            (
                np.array([0, 0, 75]),
                np.array([179, 55, 170])
            ),

            # light grey / white mould
            (
                np.array([0, 0, 170]),
                np.array([179, 45, 235])
            )
        ],

        "melon": [
            # dark bruise / rot
            # dark bruise / rot
            (
                np.array([0, 20, 0]),
                np.array([25, 255, 100])
                np.array([0, 20, 0]),
                np.array([25, 255, 100])
            ),

            # grey mould
            # grey mould
            (
                np.array([0, 0, 40]),
                np.array([179, 80, 210])
                np.array([0, 0, 40]),
                np.array([179, 80, 210])
            ),

            # light grey / white mould

            # light grey / white mould
            (
                np.array([0, 0, 150]),
                np.array([179, 65, 245])
                np.array([0, 0, 150]),
                np.array([179, 65, 245])
            )
        ],

        "peach": [
            # dark bruise / rot
            # dark bruise / rot
            (
                np.array([0, 20, 0]),
                np.array([25, 255, 105])
                np.array([0, 20, 0]),
                np.array([25, 255, 105])
            ),

            # grey mould

            # grey mould
            (
                np.array([0, 0, 40]),
                np.array([179, 80, 210])
            ),

            # light grey / white mould
            (
                np.array([0, 0, 150]),
                np.array([179, 65, 245])
                np.array([0, 0, 40]),
                np.array([179, 80, 210])
            ),

            # light grey / white mould
            (
                np.array([0, 0, 150]),
                np.array([179, 65, 245])
            )
        ],

        "pear": [
            # dark bruise / rot
            # dark bruise / rot
            (
                np.array([0, 20, 0]),
                np.array([25, 255, 110])
            ),

            # grey mould
            (
                np.array([0, 0, 40]),
                np.array([179, 80, 210])
            ),

            # light mould
            (
                np.array([0, 0, 150]),
                np.array([179, 65, 245])
            )
        ],

        "pineapple": [
            # dark / brown damaged areas
            (
                np.array([0, 20, 0]),
                np.array([25, 255, 110])
                np.array([0, 20, 0]),
                np.array([25, 255, 110])
            ),

            # grey mould
            (
                np.array([0, 0, 40]),
                np.array([179, 80, 210])
            ),

            # light mould
            (
                np.array([0, 0, 150]),
                np.array([179, 65, 245])
            )
        ],

        "pineapple": [
            # dark / brown damaged areas
            (
                np.array([0, 20, 0]),
                np.array([25, 255, 110])
            ),

            # grey mould

            # grey mould
            (
                np.array([0, 0, 40]),
                np.array([179, 80, 210])
            ),

            # light grey / white mould
            (
                np.array([0, 0, 150]),
                np.array([179, 65, 245])
            )
        ],

        "watermelon": [
            # dark damaged areas
            (
                np.array([0, 20, 0]),
                np.array([25, 255, 100])
            ),

            # grey mould
            (
                np.array([0, 0, 40]),
                np.array([179, 80, 210])
            ),

            # light grey / white mould
            (
                np.array([0, 0, 150]),
                np.array([179, 65, 245])
            )
                np.array([0, 0, 40]),
                np.array([179, 80, 210])
            ),

            # light grey / white mould
            (
                np.array([0, 0, 150]),
                np.array([179, 65, 245])
            )
        ],

        "watermelon": [
            # dark damaged areas
            (
                np.array([0, 20, 0]),
                np.array([25, 255, 100])
            ),

            # grey mould
            (
                np.array([0, 0, 40]),
                np.array([179, 80, 210])
            ),

            # light grey / white mould
            (
                np.array([0, 0, 150]),
                np.array([179, 65, 245])
            )
        ]
    }

    return fruit_hsv_ranges.get(
        fruit_type,
        default_ranges
    )
    return fruit_hsv_ranges.get(
        fruit_type,
        default_ranges
    )


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

    # Find blemish pixels
    mask_indices = blemish_mask > 0

    # If no blemish was detected, just return original image
    if not np.any(mask_indices):
        return output

    purple_layer = np.zeros_like(output)
    purple_layer[:] = (255, 0, 255)

    blended = cv2.addWeighted(
        output,
    mask_indices = blemish_mask > 0

    # No blemish pixels found — nothing to overlay, return image as-is
    if not np.any(mask_indices):
        return output

    red_layer = np.zeros_like(output)
    red_layer[:] = (0, 0, 255)

    # If no blemish was detected, just return original image
    if not np.any(mask_indices):
        return output

    purple_layer = np.zeros_like(output)
    purple_layer[:] = (255, 0, 255)

    blended = cv2.addWeighted(
        output,
        0.35,
        purple_layer,
        purple_layer,
        0.65,
        0
    )

    output[mask_indices] = blended[
        mask_indices
    ]

    output[mask_indices] = blended[
        mask_indices
    ]

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

    cv2.imshow(
        "DEBUG - Fruit Mask Used by Blemish",
        fruit_mask_binary
    )

    # ========================================================
    # FILL INTERNAL HOLES IN FRUIT MASK
    # ========================================================
    # Add black border so flood fill always starts
    # from background
    padded_mask = cv2.copyMakeBorder(
        fruit_mask_binary,
        1,
        1,
        1,
        1,
        cv2.BORDER_CONSTANT,
        value=0
    )

    flood_filled = padded_mask.copy()

    flood_mask = np.zeros(
        (
            padded_mask.shape[0] + 2,
            padded_mask.shape[1] + 2
        ),
        dtype=np.uint8
    )

    # Fill the outside background
    cv2.floodFill(
        flood_filled,
        flood_mask,
        (0, 0),
        255
    )

    # Invert to obtain only internal holes
    internal_holes = cv2.bitwise_not(
        flood_filled
    )

    # Add internal holes back into fruit
    filled_mask = cv2.bitwise_or(
        padded_mask,
        internal_holes
    )

    # Remove temporary border
    fruit_mask_binary = filled_mask[
        1:-1,
        1:-1
    ]

    # Slightly shrink fruit mask to avoid
    # dark boundary pixels being detected as blemishes
    erosion_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    fruit_mask_binary = cv2.erode(
        fruit_mask_binary,
        erosion_kernel,
        iterations=1
    )

    cv2.imshow(
        "DEBUG - Fruit Mask Used by Blemish",
        fruit_mask_binary
    )

    # ========================================================
    # FILL INTERNAL HOLES IN FRUIT MASK
    # ========================================================
    # Add black border so flood fill always starts
    # from background
    padded_mask = cv2.copyMakeBorder(
        fruit_mask_binary,
        1,
        1,
        1,
        1,
        cv2.BORDER_CONSTANT,
        value=0
    )

    flood_filled = padded_mask.copy()

    flood_mask = np.zeros(
        (
            padded_mask.shape[0] + 2,
            padded_mask.shape[1] + 2
        ),
        dtype=np.uint8
    )

    # Fill the outside background
    cv2.floodFill(
        flood_filled,
        flood_mask,
        (0, 0),
        255
    )

    # Invert to obtain only internal holes
    internal_holes = cv2.bitwise_not(
        flood_filled
    )

    # Add internal holes back into fruit
    filled_mask = cv2.bitwise_or(
        padded_mask,
        internal_holes
    )

    # Remove temporary border
    fruit_mask_binary = filled_mask[
        1:-1,
        1:-1
    ]

    # Slightly shrink fruit mask to avoid
    # dark boundary pixels being detected as blemishes
    erosion_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    fruit_mask_binary = cv2.erode(
        fruit_mask_binary,
        erosion_kernel,
        iterations=1
    )

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

    cv2.imshow(
        "DEBUG - HSV Blemish Before Cleanup",
        blemish_mask
    )

    cv2.imshow(
        "DEBUG - HSV Blemish Before Cleanup",
        blemish_mask
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

#    # ========================================================
#    # BLEMISH REVIEW STATUS
#    # ========================================================
#    if blemish_percentage < 1.0:
#        review_status = (
#            "No Significant Blemish Detected"
#        )
#
#    elif blemish_percentage <= 15.0:
#        review_status = (
#            "Review Required"
#        )
#
#    else:
#        review_status = (
#            "Significant Blemish Detected"
#        )

#    # ========================================================
#    # BLEMISH REVIEW STATUS
#    # ========================================================
#    if blemish_percentage < 1.0:
#        review_status = (
#            "No Significant Blemish Detected"
#        )
#
#    elif blemish_percentage <= 15.0:
#        review_status = (
#            "Review Required"
#        )
#
#    else:
#        review_status = (
#            "Significant Blemish Detected"
#        )

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
        "blemish_percentage": blemish_percentage,
        #"review_status": review_status
        "blemish_percentage": blemish_percentage,
        #"review_status": review_status
    }
