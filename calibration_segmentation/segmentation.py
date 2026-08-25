import cv2


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

