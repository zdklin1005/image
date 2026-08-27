import cv2
from tkinter import Tk, filedialog

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

from fruit_ripeness_object_detection.detection import (
    detect_fruit_ripeness,
    draw_detections
)

from fruit_ripeness_object_detection.blemish import (
    detect_fruit_blemish
)

def resize_for_display(image, max_width=1200, max_height=850):
    height, width = image.shape[:2]

    scale = min(
        max_width / width,
        max_height / height
    )

    new_width = int(width * scale)
    new_height = int(height * scale)

    return cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_LINEAR
    )

def run_fruit_assessment(
    image_path,
    calibration_mode=False,
    reference_width_cm=None,
    reference_height_cm=None,
    target_pixels_per_cm=20,
    use_watershed=False
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

    classification_image = preprocessing_results[
        "classification_image"
    ]

    blur_score = preprocessing_results[
        "blur_score"
    ]

    is_blurry = preprocessing_results[
        "is_blurry"
    ]

    mean_brightness = preprocessing_results[
        "mean_brightness"
    ]

    contrast_score = preprocessing_results[
        "contrast_score"
    ]

    dynamic_range = preprocessing_results[
        "dynamic_range"
    ]

    dark_pixel_ratio = preprocessing_results[
        "dark_pixel_ratio"
    ]

    bright_pixel_ratio = preprocessing_results[
        "bright_pixel_ratio"
    ]

    resize_scale = preprocessing_results[
        "resize_scale"
    ]

    resize_padding = preprocessing_results[
        "resize_padding"
    ]

    output_size = preprocessing_results[
        "output_size"
    ]

    print("\nImage Preprocessing")
    print("------------------------------")

    print(
        f"Blur score : "
        f"{blur_score:.2f}"
    )

    print(
        f"Mean brightness : "
        f"{mean_brightness:.2f}"
    )

    print(
        f"Contrast score  : "
        f"{contrast_score:.2f}"
    )

    print(
        f"Dynamic range   : "
        f"{dynamic_range:.2f}"
    )

    print(
        f"Dark pixels     : "
        f"{dark_pixel_ratio:.2%}"
    )

    print(
        f"Bright pixels   : "
        f"{bright_pixel_ratio:.2%}"
    )

    print(
        "Median filter   : Applied (5 x 5)"
    )

    print(
        "Bilateral filter: Applied"
    )

    print(
        f"Output size     : "
        f"{output_size[0]} x {output_size[1]}"
    )

    print(
        f"Resize scale    : "
        f"{resize_scale:.4f}"
    )

    print(
        "Resize padding  : "
        f"{resize_padding}"
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
    # MEMBER 3: FRUIT DETECTION AND RIPENESS CLASSIFICATION
    # ========================================================

    segmented_fruit_image = cv2.bitwise_and(
        working_image,
        working_image,
        mask=fruit_mask
    )

    detections = detect_fruit_ripeness(
        segmented_fruit_image,
        confidence_threshold=0.40
    )

    detection_image = draw_detections(
        segmented_fruit_image,
        detections
    )

    print("\nFruit Detection and Ripeness")
    print("------------------------------")

    if len(detections) == 0:
        print("No fruit detected.")

    else:
        for index, detection in enumerate(
            detections,
            start=1
        ):

            print(
                f"Detection {index}"
            )

            print(
                f"Fruit      : "
                f"{detection['fruit_type']}"
            )

            print(
                f"Ripeness   : "
                f"{detection['ripeness']}"
            )

            print(
                f"Confidence : "
                f"{detection['confidence'] * 100:.2f}%"
            )

            print(
                f"Bounding box : "
                f"{detection['bounding_box']}\n"
            )

    # ========================================================
    # MEMBER 3: BLEMISH ANALYSIS
    # ========================================================
    blemish_results = None

    if len(detections) == 0:

        print("\nBlemish Analysis")
        print("------------------------------")
        print("Skipped - no detected fruit class available.")

    else:
        # Use the highest-confidence detection
        primary_detection = max(
            detections,
            key=lambda detection: detection["confidence"]
        )

        primary_fruit_type = primary_detection[
            "fruit_type"
        ]

        blemish_results = detect_fruit_blemish(
            image=working_image,
            fruit_mask=fruit_mask,
            fruit_type=primary_fruit_type,
            opening_kernel_size=3,
            closing_kernel_size=5,
            min_component_area=60
        )

        print("\nBlemish Analysis")
        print("------------------------------")
        print(
            f"Fruit type used    : "
            f"{blemish_results['fruit_type_used']}"
        )
        print(
            f"Blemish area       : "
            f"{blemish_results['blemish_area_pixels']} pixels^2"
        )
        print(
            f"Blemish Percentage : "
            f"{blemish_results['blemish_percentage']:.2f}%"
        )

    # ========================================================
    # RETURN RESULTS
    # ========================================================

    return {
    "analysis_image": analysis_image,
    "classification_image": classification_image,
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
    "is_blurry": is_blurry,

    "mean_brightness": mean_brightness,
    "contrast_score": contrast_score,
    "dynamic_range": dynamic_range,
    "dark_pixel_ratio": dark_pixel_ratio,
    "bright_pixel_ratio": bright_pixel_ratio,

    "resize_scale": resize_scale,
    "resize_padding": resize_padding,
    "output_size": output_size,

    "detections": detections,
    "detection_image": detection_image,

    "blemish_mask": (
        blemish_results["blemish_mask"]
        if blemish_results is not None else None
    ),

    "blemish_overlay": (
        blemish_results["blemish_overlay"]
        if blemish_results is not None else None
    ),

    "blemish_area_pixels": (
        blemish_results["blemish_area_pixels"]
        if blemish_results is not None else 0
    ),

    "blemish_percentage": (
        blemish_results["blemish_percentage"]
        if blemish_results is not None else 0.0
    ),
}


if __name__ == "__main__":

    # Select Image
    root = Tk()
    root.withdraw()

    try:
        image_path = filedialog.askopenfilename(
            title="Select Fruit Image",
            filetypes=[
                ("Image Files", "*.jpg *.jpeg *.png *.bmp"),
                ("All Files", "*.*")
            ]
        )
    finally:
        root.destroy()

    # User closes the file picker without selecting an image
    if not image_path:
        print("No image selected.")
        raise SystemExit(0)

    print(f"\nSelected image: {image_path}")


    # Run complete pipeline
    results = run_fruit_assessment(
        image_path=image_path,

        # Kaggle image:
        calibration_mode=False
    )

    display_results(results)
    save_results(results)

    print("\nPress any key on an image window to continue to Fruit Ripeness Object Detection.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Fruit ripeness detection
    display_detection_image = resize_for_display(
        results["detection_image"]
    )

    cv2.namedWindow(
        "Fruit Detection and Ripeness",
        cv2.WINDOW_NORMAL
    )

    height, width = display_detection_image.shape[:2]

    cv2.resizeWindow(
        "Fruit Detection and Ripeness",
        width,
        height
    )

    cv2.imshow(
        "Fruit Detection and Ripeness",
        display_detection_image
    )

    # Blemish calculation
    if results["blemish_mask"] is not None:
        cv2.namedWindow(
            "Blemish Detection",
            cv2.WINDOW_NORMAL
        )

        cv2.imshow(
            "Blemish Detection",
            results["blemish_mask"]
        )

        cv2.namedWindow(
            "Blemish Overlay",
            cv2.WINDOW_NORMAL
        )

        cv2.imshow(
            "Blemish Overlay",
            results["blemish_overlay"]
        )

        print(
            f"\nBlemish Percentage: "
            f"{results['blemish_percentage']:.2f}%"
        )

        print("\nPress any key on an image window to exit the program.")

        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
#    print("\nPress any key on an image window to exit the program.")
#
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
