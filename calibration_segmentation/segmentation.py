import cv2
import numpy as np


def segment_fruit_otsu(image):
    """
    Segment the fruit foreground from the background
    using Otsu's automatic thresholding.

    Two approaches are tested:
    1. Grayscale Otsu
    2. HSV Saturation-channel Otsu

    Parameters:
        image:
            BGR input image.

    Returns:
        gray_image:
            Grayscale representation of the image.

        gray_mask:
            Binary mask produced using grayscale Otsu.

        gray_threshold:
            Threshold automatically selected by grayscale Otsu.

        saturation_image:
            Saturation channel extracted from HSV image.

        saturation_mask:
            Binary mask produced using saturation-channel Otsu.

        saturation_threshold:
            Threshold automatically selected by saturation Otsu.
    """

    if image is None:
        raise ValueError("Input image cannot be None.")

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input image must be a three-channel BGR image.")

    # ============================================================
    # Method 1: Grayscale Otsu
    # ============================================================

    gray_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Inverted because:
    # darker fruit      -> white foreground
    # bright background -> black background
    gray_threshold, gray_mask = cv2.threshold(
        gray_image,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # ============================================================
    # Method 2: HSV Saturation-channel Otsu
    # ============================================================

    hsv_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    # Extract Saturation (S) channel
    saturation_image = hsv_image[:, :, 1]

    # Normal binary is used because:
    # colourful/saturated fruit      -> white foreground
    # low-saturation white background -> black background
    saturation_threshold, saturation_mask = cv2.threshold(
        saturation_image,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return (
        gray_image,
        gray_mask,
        gray_threshold,
        saturation_image,
        saturation_mask,
        saturation_threshold
    )

def combine_otsu_masks_constrained(
    gray_mask,
    saturation_mask,
    expansion_kernel_size=9
):
    """
    Combine grayscale and saturation Otsu masks while
    preventing large unrelated grayscale regions, such as
    shadows, from being added to the fruit mask.

    The saturation mask is treated as the primary fruit
    region. Grayscale foreground is only accepted if it lies
    within a small expanded neighbourhood around the
    saturation foreground.

    Parameters:
        gray_mask:
            Binary mask from grayscale Otsu.

        saturation_mask:
            Binary mask from saturation-channel Otsu.

        expansion_kernel_size:
            Odd kernel size used to expand the saturation
            region before allowing grayscale recovery.

    Returns:
        combined_mask:
            Constrained combined fruit mask.
    """

    if gray_mask is None or saturation_mask is None:
        raise ValueError(
            "Input masks cannot be None."
        )

    if gray_mask.shape != saturation_mask.shape:
        raise ValueError(
            "Input masks must have the same dimensions."
        )

    if (
        expansion_kernel_size <= 0
        or expansion_kernel_size % 2 == 0
    ):
        raise ValueError(
            "expansion_kernel_size must be a positive odd integer."
        )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            expansion_kernel_size,
            expansion_kernel_size
        )
    )

    # Expand the main saturation-based fruit region slightly.
    expanded_saturation = cv2.dilate(
        saturation_mask,
        kernel,
        iterations=1
    )

    # Only allow grayscale foreground that is close to the
    # saturation-based fruit region.
    recovered_gray = cv2.bitwise_and(
        gray_mask,
        expanded_saturation
    )

    # Combine the original saturation mask with the accepted
    # grayscale recovery regions.
    combined_mask = cv2.bitwise_or(
        saturation_mask,
        recovered_gray
    )

    return combined_mask

def refine_fruit_mask(
    mask,
    opening_kernel_size=3,
    closing_kernel_size=5
):
    """
    Refine a binary fruit mask using morphological
    opening and closing.

    Opening:
        Removes small white foreground noise.

    Closing:
        Fills small black holes and gaps inside
        the fruit region.

    Parameters:
        mask:
            Binary input mask.

        opening_kernel_size:
            Kernel size used for morphological opening.

        closing_kernel_size:
            Kernel size used for morphological closing.

    Returns:
        opened_mask:
            Result after morphological opening.

        refined_mask:
            Final result after opening followed by closing.
    """

    if mask is None:
        raise ValueError("Input mask cannot be None.")

    # Validate opening kernel
    if (
        opening_kernel_size <= 0
        or opening_kernel_size % 2 == 0
    ):
        raise ValueError(
            "opening_kernel_size must be a positive odd integer."
        )

    # Validate closing kernel
    if (
        closing_kernel_size <= 0
        or closing_kernel_size % 2 == 0
    ):
        raise ValueError(
            "closing_kernel_size must be a positive odd integer."
        )

    # ============================================================
    # Create separate structuring elements
    # ============================================================

    opening_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            opening_kernel_size,
            opening_kernel_size
        )
    )

    closing_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            closing_kernel_size,
            closing_kernel_size
        )
    )

    # ============================================================
    # Morphological Opening
    # Erosion followed by dilation
    # Removes small white foreground noise
    # ============================================================

    opened_mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        opening_kernel
    )

    # ============================================================
    # Morphological Closing
    # Dilation followed by erosion
    # Fills small black holes and gaps
    # ============================================================

    refined_mask = cv2.morphologyEx(
        opened_mask,
        cv2.MORPH_CLOSE,
        closing_kernel
    )

    return opened_mask, refined_mask


def prepare_watershed_mask(refined_mask):
    """
    Prepare a solid foreground mask specifically for
    Watershed segmentation.

    Internal holes in detected fruit regions are filled so
    that the distance transform represents the overall fruit
    shapes rather than colour or highlight variations.

    Parameters:
        refined_mask:
            Binary mask after morphological refinement.

    Returns:
        watershed_mask:
            Solid binary foreground mask for Watershed.
    """

    if refined_mask is None:
        raise ValueError(
            "Refined mask cannot be None."
        )

    # Ensure proper binary mask
    binary_mask = np.where(
        refined_mask > 0,
        255,
        0
    ).astype(np.uint8)

    # Find only external foreground contours
    contours, _ = cv2.findContours(
        binary_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        raise ValueError(
            "No foreground region was found."
        )

    watershed_mask = np.zeros_like(
        binary_mask
    )

    # Find largest foreground area
    largest_area = max(
        cv2.contourArea(contour)
        for contour in contours
    )

    # Ignore very small foreground noise
    minimum_area = 0.05 * largest_area

    # Fill meaningful foreground objects completely
    for contour in contours:

        if cv2.contourArea(contour) >= minimum_area:

            cv2.drawContours(
                watershed_mask,
                [contour],
                -1,
                255,
                thickness=cv2.FILLED
            )

    return watershed_mask

def apply_watershed_segmentation(
    image,
    refined_mask,
    foreground_ratio=0.4
):
    """
    Apply marker-based Watershed segmentation to separate
    touching foreground fruits.

    Parameters:
        image:
            BGR colour image corresponding to the mask.

        refined_mask:
            Binary foreground mask after morphological
            refinement.

        foreground_ratio:
            Fraction of the maximum distance-transform value
            used to determine sure foreground regions.

    Returns:
        watershed_markers:
            Label image produced by Watershed.

            -1 = watershed boundary
             1 = background
            >1 = individual foreground objects

        separated_mask:
            Binary mask containing all detected fruit regions.

        distance_transform:
            Distance-transform image used to identify
            object centres.

        sure_foreground:
            Binary sure-foreground marker image.

        fruit_labels:
            List of individual fruit labels.
    """

    if image is None:
        raise ValueError(
            "Input image cannot be None."
        )

    if refined_mask is None:
        raise ValueError(
            "Refined mask cannot be None."
        )

    if image.shape[:2] != refined_mask.shape[:2]:
        raise ValueError(
            "Image and refined mask must have "
            "the same dimensions."
        )

    if not 0 < foreground_ratio < 1:
        raise ValueError(
            "foreground_ratio must be between 0 and 1."
        )

    # ========================================================
    # Prepare Watershed input mask
    # ========================================================

    binary_mask = prepare_watershed_mask(
        refined_mask
    )

    # ========================================================
    # Step 1: Sure background
    # ========================================================

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )

    sure_background = cv2.dilate(
        binary_mask,
        kernel,
        iterations=3
    )

    # ========================================================
    # Step 2: Distance transform
    # ========================================================

    distance_transform = cv2.distanceTransform(
        binary_mask,
        cv2.DIST_L2,
        5
    )

    max_distance = distance_transform.max()

    if max_distance <= 0:
        raise ValueError(
            "No foreground region was found in the mask."
        )

    # ========================================================
    # Step 3: Sure foreground
    # ========================================================

    _, sure_foreground = cv2.threshold(
        distance_transform,
        foreground_ratio * max_distance,
        255,
        cv2.THRESH_BINARY
    )

    sure_foreground = sure_foreground.astype(
        np.uint8
    )

    # ========================================================
    # Step 4: Unknown region
    # ========================================================

    unknown_region = cv2.subtract(
        sure_background,
        sure_foreground
    )

    # ========================================================
    # Step 5: Connected-component markers
    # ========================================================

    _, markers = cv2.connectedComponents(
        sure_foreground
    )

    # Marker 1 represents known background.
    # Individual objects begin from marker 2.
    markers = markers + 1

    # Unknown regions are set to marker 0.
    markers[
        unknown_region == 255
    ] = 0

    # ========================================================
    # Step 6: Watershed
    # ========================================================

    watershed_markers = cv2.watershed(
        image.copy(),
        markers
    )

    # ========================================================
    # Step 7: Determine individual fruit labels
    # ========================================================

    unique_labels = np.unique(
        watershed_markers
    )

    fruit_labels = [
        int(label)
        for label in unique_labels
        if label > 1
    ]

    # ========================================================
    # Step 8: Combined binary fruit mask
    # ========================================================

    separated_mask = np.zeros_like(
        binary_mask
    )

    separated_mask[
        watershed_markers > 1
    ] = 255

    return (
        watershed_markers,
        separated_mask,
        distance_transform,
        sure_foreground,
        fruit_labels
    )
