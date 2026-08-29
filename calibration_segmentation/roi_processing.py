import cv2

from calibration_segmentation.segmentation import (
    segment_fruit_otsu,
    combine_otsu_masks_constrained,
    refine_fruit_mask,
    apply_watershed_segmentation,
    select_watershed_target_region
)

from calibration_segmentation.measurement import (
    extract_main_fruit,
)

from calibration_segmentation.feature_extraction import (
    extract_colour_features
)


def process_fruit_roi(image, bounding_box, use_watershed=False, global_refined_mask=None):
    """
    Process one YOLO-detected fruit ROI using:
    Otsu segmentation
    -> separate mask refinement
    -> constrained combination
    -> final refinement
    -> fruit extraction
    -> colour feature extraction
    """

    if image is None:
        raise ValueError("Input image cannot be None.")

    if bounding_box is None:
        raise ValueError("Bounding box cannot be None.")

    image_height, image_width = image.shape[:2]

    x1, y1, x2, y2 = map(
        int,
        bounding_box
    )

    # Keep coordinates inside image boundaries
    x1 = max(0, min(x1, image_width - 1))
    y1 = max(0, min(y1, image_height - 1))

    x2 = max(0, min(x2, image_width))
    y2 = max(0, min(y2, image_height))

    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            "Invalid ROI bounding box."
        )

    # ----------------------------------------------------
    # 1. Extract ROI
    # ----------------------------------------------------

    roi_image = image[
        y1:y2,
        x1:x2
    ].copy()

    global_roi_mask = None

    if global_refined_mask is not None:

        global_roi_mask = global_refined_mask[
            y1:y2,
            x1:x2
        ].copy()

        if global_roi_mask.size == 0:
            global_roi_mask = None

    # ----------------------------------------------------
    # 2. Otsu segmentation
    # ----------------------------------------------------

    (
        gray_image,
        gray_mask,
        gray_threshold,
        saturation_image,
        saturation_mask,
        saturation_threshold
    ) = segment_fruit_otsu(
        roi_image
    )

    # ----------------------------------------------------
    # 3. Refine grayscale mask separately
    # ----------------------------------------------------

    (
        gray_opened_mask,
        gray_refined_mask
    ) = refine_fruit_mask(
        gray_mask,
        opening_kernel_size=3,
        closing_kernel_size=5
    )

    # ----------------------------------------------------
    # 4. Refine saturation mask separately
    # ----------------------------------------------------

    (
        saturation_opened_mask,
        saturation_refined_mask
    ) = refine_fruit_mask(
        saturation_mask,
        opening_kernel_size=3,
        closing_kernel_size=5
    )

    # ----------------------------------------------------
    # 5. Combine refined masks
    # ----------------------------------------------------

    combined_mask = combine_otsu_masks_constrained(
        gray_refined_mask,
        saturation_refined_mask,
        expansion_kernel_size=9
    )

    # ----------------------------------------------------
    # 6. Final refinement
    # ----------------------------------------------------

    (
        opened_mask,
        refined_mask
    ) = refine_fruit_mask(
        combined_mask,
        opening_kernel_size=3,
        closing_kernel_size=5
    )

    if use_watershed:

        (
            watershed_markers,
            separated_mask,
            distance_transform,
            sure_foreground,
            fruit_labels
        ) = apply_watershed_segmentation(
            roi_image,
            refined_mask,
            foreground_ratio=0.4
        )

        if len(fruit_labels) > 1:

            (
                final_processing_mask,
                selected_watershed_label
            ) = select_watershed_target_region(
                watershed_markers,
                fruit_labels
            )

        else:

            # Watershed did not actually separate
            # multiple objects.
            final_processing_mask = refined_mask

            selected_watershed_label = None

    else:

        watershed_markers = None
        separated_mask = None
        distance_transform = None
        sure_foreground = None
        fruit_labels = None

        selected_watershed_label = None

        final_processing_mask = refined_mask
    # ----------------------------------------------------
    # 7. Extract main fruit in ROI
    # ----------------------------------------------------
    
    if global_roi_mask is not None:

        final_processing_mask = global_roi_mask

    (
        fruit_mask,
        fruit_contour,
        fruit_area_pixels
    ) = extract_main_fruit(
        final_processing_mask
    )


    # ----------------------------------------------------
    # 8. Preserve fruit colour
    # ----------------------------------------------------

    fruit_only_colour = cv2.bitwise_and(
        roi_image,
        roi_image,
        mask=fruit_mask
    )

    # ----------------------------------------------------
    # 9. Extract colour features
    # ----------------------------------------------------

    colour_features = extract_colour_features(
        roi_image,
        fruit_mask
    )

    return {
        "bounding_box": (
            x1,
            y1,
            x2,
            y2
        ),

        "roi_image": roi_image,

        "watershed_markers": watershed_markers,
        "separated_mask": separated_mask,
        "distance_transform": distance_transform,
        "sure_foreground": sure_foreground,
        "fruit_labels": fruit_labels,
        "selected_watershed_label": selected_watershed_label,
        "watershed_used": use_watershed,

        "gray_image": gray_image,
        "gray_mask": gray_mask,
        "gray_opened_mask": gray_opened_mask,
        "gray_refined_mask": gray_refined_mask,
        "gray_threshold": gray_threshold,

        "saturation_image": saturation_image,
        "saturation_mask": saturation_mask,
        "saturation_opened_mask": saturation_opened_mask,
        "saturation_refined_mask": saturation_refined_mask,
        "saturation_threshold": saturation_threshold,

        "combined_mask": combined_mask,

        "opened_mask": opened_mask,
        "refined_mask": refined_mask,

        "fruit_mask": fruit_mask,
        "fruit_contour": fruit_contour,

        "fruit_only_colour": fruit_only_colour,

        "fruit_area_pixels": fruit_area_pixels,

        "colour_features": colour_features,

        "global_roi_mask": global_roi_mask,

    }

