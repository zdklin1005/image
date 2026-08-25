import cv2
import numpy as np


def extract_main_fruit(
    refined_mask,
    stem_removal_kernel_size=5
):
    """
    Extract the main fruit body from the refined binary mask
    for projected area measurement.

    A small morphological opening is applied before contour
    extraction to remove thin protrusions such as the fruit
    stem without changing the Technique 5 refined mask.

    Parameters:
        refined_mask:
            Binary mask after morphological refinement.

        stem_removal_kernel_size:
            Kernel size used to remove thin connected
            structures such as the stem.

    Returns:
        fruit_mask:
            Filled binary mask containing only the main
            fruit body.

        fruit_contour:
            External contour of the selected fruit body.

        fruit_area_pixels:
            Projected fruit area in pixels.
    """

    # --------------------------------------------------------
    # Validate kernel size
    # --------------------------------------------------------

    if (
        stem_removal_kernel_size <= 0
        or stem_removal_kernel_size % 2 == 0
    ):
        raise ValueError(
            "stem_removal_kernel_size must be "
            "a positive odd integer."
        )

    # --------------------------------------------------------
    # Create elliptical kernel
    # --------------------------------------------------------

    stem_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            stem_removal_kernel_size,
            stem_removal_kernel_size
        )
    )

    # --------------------------------------------------------
    # Remove thin protrusions such as the stem
    # --------------------------------------------------------

    measurement_mask = cv2.morphologyEx(
        refined_mask,
        cv2.MORPH_OPEN,
        stem_kernel
    )

    # --------------------------------------------------------
    # Find external foreground contours
    # --------------------------------------------------------

    contours, _ = cv2.findContours(
        measurement_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        raise ValueError(
            "No fruit foreground was detected."
        )

    # --------------------------------------------------------
    # Select the largest foreground object
    # --------------------------------------------------------

    fruit_contour = max(
        contours,
        key=cv2.contourArea
    )

    # --------------------------------------------------------
    # Create final solid fruit-body mask
    # --------------------------------------------------------

    fruit_mask = np.zeros_like(
        refined_mask
    )

    cv2.drawContours(
        fruit_mask,
        [fruit_contour],
        -1,
        255,
        thickness=cv2.FILLED
    )

    # --------------------------------------------------------
    # Calculate projected area in pixels
    # --------------------------------------------------------

    fruit_area_pixels = cv2.countNonZero(
        fruit_mask
    )

    return (
        fruit_mask,
        fruit_contour,
        fruit_area_pixels
    )

def calculate_projected_area_cm2(
    fruit_area_pixels,
    pixels_per_cm_x,
    pixels_per_cm_y
):
    """
    Convert the projected fruit area from pixels
    into square centimetres using spatial calibration.

    Parameters:
        fruit_area_pixels:
            Number of foreground pixels representing
            the fruit body.

        pixels_per_cm_x:
            Horizontal spatial scale in pixels/cm.

        pixels_per_cm_y:
            Vertical spatial scale in pixels/cm.

    Returns:
        fruit_area_cm2:
            Estimated projected fruit area in cm^2.
    """

    # Validate spatial calibration values
    if pixels_per_cm_x <= 0 or pixels_per_cm_y <= 0:
        raise ValueError(
            "Pixels-per-centimetre values must be greater than zero."
        )

    # Convert pixel area into physical projected area
    fruit_area_cm2 = (
        fruit_area_pixels
        / (pixels_per_cm_x * pixels_per_cm_y)
    )

    return fruit_area_cm2