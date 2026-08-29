import cv2
import numpy as np


def extract_main_fruit(
    refined_mask,
    cleanup_kernel_size=5
):
    """
    Extract the main fruit body from the refined binary mask
    for projected area measurement.

    A morphological opening is applied before contour
    extraction to remove thin connected structures such as
    stems, narrow shadow connections, and small protrusions.

    Parameters:
        refined_mask:
            Binary mask after morphological refinement.

        cleanup_kernel_size:
            Kernel size used for measurement-specific
            foreground cleanup.

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
    # Validate input
    # --------------------------------------------------------

    if refined_mask is None:
        raise ValueError(
            "Refined mask cannot be None."
        )

    if (
        cleanup_kernel_size <= 0
        or cleanup_kernel_size % 2 == 0
    ):
        raise ValueError(
            "cleanup_kernel_size must be "
            "a positive odd integer."
        )

    # --------------------------------------------------------
    # Ensure proper binary mask
    # --------------------------------------------------------

    binary_mask = np.where(
        refined_mask > 0,
        255,
        0
    ).astype(np.uint8)

    # --------------------------------------------------------
    # Create elliptical cleanup kernel
    # --------------------------------------------------------

    cleanup_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            cleanup_kernel_size,
            cleanup_kernel_size
        )
    )

    # --------------------------------------------------------
    # Remove narrow connected structures
    # --------------------------------------------------------

    measurement_mask = cv2.morphologyEx(
        binary_mask,
        cv2.MORPH_OPEN,
        cleanup_kernel
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
    # Select largest foreground object
    # --------------------------------------------------------

    fruit_contour = max(
        contours,
        key=cv2.contourArea
    )

    if cv2.contourArea(fruit_contour) <= 0:
        raise ValueError(
            "Detected fruit contour has zero area."
        )

    # --------------------------------------------------------
    # Create final solid fruit mask
    # --------------------------------------------------------

    fruit_mask = np.zeros_like(
        binary_mask
    )

    cv2.drawContours(
        fruit_mask,
        [fruit_contour],
        -1,
        255,
        thickness=cv2.FILLED
    )

    # --------------------------------------------------------
    # Calculate projected fruit area
    # --------------------------------------------------------

    fruit_area_pixels = cv2.countNonZero(
        fruit_mask
    )

    return (
        fruit_mask,
        fruit_contour,
        fruit_area_pixels
    )

