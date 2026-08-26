import cv2
import os

from preprocessing import (
    preprocess_fruit_image
)

from calibration_segmentation.calibration import (
    calibrate_image
)

from calibration_segmentation.segmentation import (
    segment_fruit_otsu,
    combine_otsu_masks_constrained,
    refine_fruit_mask,
    apply_watershed_segmentation
)

from calibration_segmentation.measurement import (
    extract_main_fruit,
    calculate_projected_area_cm2
)

from calibration_segmentation.visualisation import (
    display_results,
    save_results
)

def run_fruit_assessment(
    image_path,
    calibration_mode=False,
    reference_width_cm=None,
    reference_height_cm=None,
    target_pixels_per_cm=20
):
    """
    Run the integrated fruit image-processing pipeline.
    """

    # ========================================================
    # MEMBER 1: PREPROCESSING
    # ========================================================

    preprocessing_results = (
        preprocess_fruit_image(
            image_path
        )
    )

    analysis_image = preprocessing_results[
        "analysis_image"
    ]

    display_image = preprocessing_results[
        "display_image"
    ]

    blur_score = preprocessing_results[
        "blur_score"
    ]

    is_blurry = preprocessing_results[
        "is_blurry"
    ]

    print("\nImage Preprocessing")
    print("------------------------------")

    print(
        f"Blur score : "
        f"{blur_score:.2f}"
    )

    if is_blurry:
        print(
            "Warning: Image may be too blurry."
        )
    else:
        print(
            "Image sharpness is acceptable."
        )


    # ========================================================
    # MEMBER 2: CALIBRATION
    # ========================================================

    if calibration_mode:

        if (
            reference_width_cm is None
            or reference_height_cm is None
        ):
            raise ValueError(
                "Reference dimensions are required "
                "when calibration_mode=True."
            )

        calibration_results = calibrate_image(
            analysis_image,
            reference_width_cm,
            reference_height_cm,
            target_pixels_per_cm,
            selection_image=display_image
        )

        working_image = calibration_results[
            "rectified_image"
        ]

        pixels_per_cm_x = calibration_results[
            "pixels_per_cm_x"
        ]

        pixels_per_cm_y = calibration_results[
            "pixels_per_cm_y"
        ]

        print("\nSpatial Calibration")
        print("------------------------------")

        print(
            f"Horizontal scale : "
            f"{pixels_per_cm_x:.2f} pixels/cm"
        )

        print(
            f"Vertical scale   : "
            f"{pixels_per_cm_y:.2f} pixels/cm"
        )

    else:

        # Dataset images normally have no known
        # physical scale.
        working_image = analysis_image.copy()

        pixels_per_cm_x = None
        pixels_per_cm_y = None

        print("\nSpatial Calibration")
        print("------------------------------")

        print(
            "Skipped - no known physical "
            "reference available."
        )


    # ========================================================
    # TECHNIQUE 4: OTSU'S BINARISATION
    # ========================================================

    (
        gray_image,
        gray_mask,
        gray_threshold,
        saturation_image,
        saturation_mask,
        saturation_threshold
    ) = segment_fruit_otsu(
        working_image
    )
    combined_mask = combine_otsu_masks_constrained(
        gray_mask,
        saturation_mask,
        expansion_kernel_size = 9
    )

    print("\nOtsu Segmentation")
    print("------------------------------")

    print(
        f"Grayscale threshold  : "
        f"{gray_threshold:.2f}"
    )

    print(
        f"Saturation threshold : "
        f"{saturation_threshold:.2f}"
    )


    # ========================================================
    # TECHNIQUE 5: MORPHOLOGICAL REFINEMENT
    # ========================================================

    opened_mask, refined_mask = (
        refine_fruit_mask(
            combined_mask,
            opening_kernel_size=3,
            closing_kernel_size=5
        )
    )


    # ========================================================
    # TECHNIQUE 6: WATERSHED
    # ========================================================

    use_watershed = False

    if use_watershed:

        (
            watershed_markers,
            separated_mask,
            distance_transform,
            sure_foreground,
            fruit_labels,
        ) = apply_watershed_segmentation(
            working_image,
            refined_mask,
            foreground_ratio=0.4
        )

        measurement_mask = separated_mask

        print("\nWatershed Segmentation")
        print("------------------------------")
        print("Watershed applied.")
        print(
            f"Number of detected fruit regions : "
            f"{len(fruit_labels)}"
        )

    else:

        watershed_markers = None
        separated_mask = None
        distance_transform = None
        sure_foreground = None
        fruit_labels = None
        measurement_mask = refined_mask

        print("\nWatershed Segmentation")
        print("------------------------------")
        print(
            "Skipped - Watershed is disabled."
        )


    # ========================================================
    # TECHNIQUE 7: FRUIT AREA MEASUREMENT
    # ========================================================

    (
        fruit_mask,
        fruit_contour,
        fruit_area_pixels
    ) = extract_main_fruit(
        measurement_mask
    )

    print("\nFruit Measurement")
    print("------------------------------")

    print(
        f"Projected fruit area : "
        f"{fruit_area_pixels} pixels^2"
    )

    fruit_area_cm2 = None

    if calibration_mode:

        fruit_area_cm2 = (
            calculate_projected_area_cm2(
                fruit_area_pixels,
                pixels_per_cm_x,
                pixels_per_cm_y
            )
        )

        print(
            f"Physical projected area : "
            f"{fruit_area_cm2:.2f} cm^2"
        )

    else:

        print(
            "Physical projected area : "
            "Not available"
        )


    # ========================================================
    # RETURN RESULTS
    # ========================================================

    return {
    "analysis_image": analysis_image,
    "working_image": working_image,

    "gray_image": gray_image,
    "gray_mask": gray_mask,

    "saturation_image": saturation_image,
    "saturation_mask": saturation_mask,

    "combined_mask": combined_mask,

    "watershed_markers": watershed_markers,
    "separated_mask": separated_mask,
    "distance_transform": distance_transform,
    "sure_foreground": sure_foreground,
    "fruit_labels": fruit_labels,


    "opened_mask": opened_mask,
    "refined_mask": refined_mask,

    "fruit_mask": fruit_mask,
    "fruit_contour": fruit_contour,

    "fruit_area_pixels": fruit_area_pixels,
    "fruit_area_cm2": fruit_area_cm2,

    "pixels_per_cm_x": pixels_per_cm_x,
    "pixels_per_cm_y": pixels_per_cm_y,

    "blur_score": blur_score,
    "is_blurry": is_blurry
}


if __name__ == "__main__":

    results = run_fruit_assessment(
        image_path="calibration_segmentation/test_images/tilt_apple.png",

        # Kaggle image:
        calibration_mode=False
    )

    display_results(results)

    save_results(results)
