import cv2
import numpy as np

def select_otsu_foreground_mask(
    normal_mask,
    inverted_mask,
    default_mask="normal"
):
    """
    Choose whether the normal or inverted Otsu mask
    is more likely to represent the detected fruit.

    Since this function is used on a YOLO fruit ROI,
    the target fruit is expected to occupy more of
    the centre than the outer border.

    Parameters
    ----------
    normal_mask : np.ndarray
        Otsu result using THRESH_BINARY.

    inverted_mask : np.ndarray
        Otsu result using THRESH_BINARY_INV.

    default_mask : str
        Fallback polarity when both candidates
        receive very similar scores.

    Returns
    -------
    selected_mask : np.ndarray
        Binary mask whose polarity is more likely
        to represent the fruit foreground.

    selected_polarity : str
        "normal" or "inverted".
    """

    if normal_mask is None or inverted_mask is None:
        raise ValueError(
            "Otsu candidate masks cannot be None."
        )

    if normal_mask.shape != inverted_mask.shape:
        raise ValueError(
            "Otsu candidate masks must have "
            "the same dimensions."
        )

    height, width = normal_mask.shape[:2]

    # --------------------------------------------------------
    # Central region
    #
    # YOLO should place the fruit approximately around
    # the centre of its bounding-box ROI.
    # --------------------------------------------------------

    centre_x1 = int(width * 0.25)
    centre_x2 = int(width * 0.75)

    centre_y1 = int(height * 0.25)
    centre_y2 = int(height * 0.75)

    # --------------------------------------------------------
    # Border region
    #
    # Outer 10% of ROI is more likely to contain background.
    # --------------------------------------------------------

    border_size_x = max(
        1,
        int(width * 0.10)
    )

    border_size_y = max(
        1,
        int(height * 0.10)
    )

    def calculate_mask_score(mask):

        foreground = mask > 0

        centre_region = foreground[
            centre_y1:centre_y2,
            centre_x1:centre_x2
        ]

        centre_ratio = np.mean(
            centre_region
        )

        # Create border selection
        border_region = np.zeros(
            foreground.shape,
            dtype=bool
        )

        border_region[
            :border_size_y,
            :
        ] = True

        border_region[
            height - border_size_y:,
            :
        ] = True

        border_region[
            :,
            :border_size_x
        ] = True

        border_region[
            :,
            width - border_size_x:
        ] = True

        border_ratio = np.mean(
            foreground[
                border_region
            ]
        )

        # Fruit should preferably be white near
        # the centre and less white at the border.
        score = (
            centre_ratio
            - border_ratio
        )

        return (
            float(score),
            float(centre_ratio),
            float(border_ratio)
        )

    (
        normal_score,
        normal_centre,
        normal_border
    ) = calculate_mask_score(
        normal_mask
    )

    (
        inverted_score,
        inverted_centre,
        inverted_border
    ) = calculate_mask_score(
        inverted_mask
    )

    # --------------------------------------------------------
    # Select polarity
    # --------------------------------------------------------

    score_difference = abs(
        normal_score - inverted_score
    )

    # If the result is ambiguous, preserve the
    # previously expected polarity.
    if score_difference < 0.05:

        if default_mask == "inverted":

            selected_mask = inverted_mask
            selected_polarity = "inverted"

        else:

            selected_mask = normal_mask
            selected_polarity = "normal"

    elif inverted_score > normal_score:

        selected_mask = inverted_mask
        selected_polarity = "inverted"

    else:

        selected_mask = normal_mask
        selected_polarity = "normal"

    return (
        selected_mask,
        selected_polarity
    )

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
    # Calculate Otsu threshold using both possible
    # foreground polarities.

    gray_threshold, gray_normal_mask = cv2.threshold(
        gray_image,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    _, gray_inverted_mask = cv2.threshold(
        gray_image,
        gray_threshold,
        255,
        cv2.THRESH_BINARY_INV
    )

    (
        gray_mask,
        gray_polarity
    ) = select_otsu_foreground_mask(
        gray_normal_mask,
        gray_inverted_mask,

        # Preserve previous behaviour when ambiguous
        default_mask="inverted"
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
    saturation_threshold, saturation_normal_mask = (
        cv2.threshold(
            saturation_image,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    )

    _, saturation_inverted_mask = cv2.threshold(
        saturation_image,
        saturation_threshold,
        255,
        cv2.THRESH_BINARY_INV
    )

    (
        saturation_mask,
        saturation_polarity
    ) = select_otsu_foreground_mask(
        saturation_normal_mask,
        saturation_inverted_mask,

        # Preserve previous behaviour when ambiguous
        default_mask="normal"
    )

    print(
        f"Grayscale Otsu polarity : "
        f"{gray_polarity}"
    )

    print(
        f"Saturation Otsu polarity: "
        f"{saturation_polarity}"
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

def select_watershed_target_region(
    watershed_markers,
    fruit_labels
):
    """
    Select the Watershed region that most likely belongs
    to the YOLO target fruit.

    Priority:
    1. Region containing the ROI centre.
    2. Otherwise, region whose centroid is closest
       to the ROI centre.
    """

    if watershed_markers is None:
        raise ValueError(
            "Watershed markers cannot be None."
        )

    if not fruit_labels:
        raise ValueError(
            "No Watershed fruit labels available."
        )

    height, width = watershed_markers.shape[:2]

    centre_x = width // 2
    centre_y = height // 2

    centre_label = int(
        watershed_markers[
            centre_y,
            centre_x
        ]
    )

    # ----------------------------------------------------
    # First choice:
    # label containing ROI centre
    # ----------------------------------------------------

    if centre_label in fruit_labels:

        target_label = centre_label

    else:

        target_label = None
        minimum_distance = None

        # ------------------------------------------------
        # Fallback:
        # choose centroid closest to ROI centre
        # ------------------------------------------------

        for label in fruit_labels:

            region_mask = (
                watershed_markers == label
            ).astype(np.uint8)

            moments = cv2.moments(
                region_mask
            )

            if moments["m00"] == 0:
                continue

            centroid_x = (
                moments["m10"]
                / moments["m00"]
            )

            centroid_y = (
                moments["m01"]
                / moments["m00"]
            )

            distance = (
                (centroid_x - centre_x) ** 2
                +
                (centroid_y - centre_y) ** 2
            )

            if (
                minimum_distance is None
                or distance < minimum_distance
            ):
                minimum_distance = distance
                target_label = label

        if target_label is None:
            raise ValueError(
                "Unable to select Watershed target region."
            )

    target_mask = np.zeros(
        watershed_markers.shape,
        dtype=np.uint8
    )

    target_mask[
        watershed_markers == target_label
    ] = 255

    return (
        target_mask,
        int(target_label)
    )
