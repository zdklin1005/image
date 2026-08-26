import cv2
import os
from tkinter import Tk, filedialog

from preprocessing import (
    preprocess_fruit_image
)

from calibration_segmentation.calibration import (
    calibrate_image
)

from calibration_segmentation.segmentation import (
    segment_fruit_otsu,
    refine_fruit_mask
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
            saturation_mask,
            opening_kernel_size=3,
            closing_kernel_size=5
        )
    )


    # ========================================================
    # TECHNIQUE 6: WATERSHED
    # ========================================================

    # Optional - implement later for touching fruits.


    # ========================================================
    # TECHNIQUE 7: FRUIT AREA MEASUREMENT
    # ========================================================

    (
        fruit_mask,
        fruit_contour,
        fruit_area_pixels
    ) = extract_main_fruit(
        refined_mask
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

    detections = detect_fruit_ripeness(
        working_image,
        confidence_threshold=0.40
    )

    detection_image = draw_detections(
        working_image,
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
    # RETURN RESULTS FOR OTHER MODULES
    # ========================================================

    return {
    "analysis_image": analysis_image,
    "working_image": working_image,

    "gray_image": gray_image,
    "gray_mask": gray_mask,

    "saturation_image": saturation_image,
    "saturation_mask": saturation_mask,

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

    "detections": detections,
    "detection_image": detection_image
}


if __name__ == "__main__":

    # Select Image
    root = Tk()
    root.withdraw()

    image_path = filedialog.askopenfilename(
        title="Select Fruit Image",
        filetypes=[
            ("Image Files", "*.jpg *.jpeg *.png *.bmp"),
            ("All Files", "*.*")
        ]
    )

    # User closes the file picker without selecting an image
    if not image_path:
        print("No image selected.")
        exit()

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
    cv2.namedWindow(
            "Fruit Detection and Ripeness",
            cv2.WINDOW_NORMAL
        )
    
    cv2.imshow(
        "Fruit Detection and Ripeness",
        results["detection_image"]
    )

    print("\nPress any key on an image window to exit the program.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()
