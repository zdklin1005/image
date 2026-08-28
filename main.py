import cv2
from tkinter import Tk, filedialog

from preprocessing import (
    preprocess_fruit_image
)

from calibration_segmentation.roi_processing import (
    process_fruit_roi
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

from calibration_segmentation.feature_extraction import (
    extract_colour_features
)

from calibration_segmentation.measurement import (
    extract_main_fruit,
    calculate_projected_area_cm2
)

from calibration_segmentation.visualisation import (
    display_results,
    display_roi_results,
    save_results
)

from fruit_ripeness_object_detection.fruit_detection import (
    detect_with_model_a,
    detect_with_model_c,
    detect_with_model_d,
    fuse_detections,
    draw_final_detections,
    crop_all_detected_fruits
)

from fruit_ripeness_object_detection.ripeness_classification import (
    classify_with_model_b,
    classify_with_model_e,
    fuse_ripeness
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
    # MEMBER 3: FRUIT DETECTION
    # ========================================================
    # Model A
    detections_a = detect_with_model_a(
        classification_image,
        confidence_threshold=0.30
    )

    # Model C
    detections_c = detect_with_model_c(
        classification_image,
        confidence_threshold=0.30
    )

    # Model D
    detections_d = detect_with_model_d(
        classification_image,
        confidence_threshold=0.30
    )

    # ========================================================
    # RAW DETECTION COUNTS
    # ========================================================
    print("\nRaw Model Detection Counts")
    print("------------------------------")

    print(
        f"Model A : "
        f"{len(detections_a)}"
    )

    print(
        f"Model C : "
        f"{len(detections_c)}"
    )

    print(
        f"Model D : "
        f"{len(detections_d)}"
    )

    # ========================================================
    # FUSE MODEL A + MODEL C + MODEL D
    # ========================================================

    final_detections = fuse_detections(
        detections_a,
        detections_c,
        detections_d,
        iou_threshold=0.30
    )

    # ========================================================
    # FRUIT DETECTION RESULTS
    # ========================================================
    print("\nFruit Detection")
    print("------------------------------")

    if len(final_detections) == 0:

        print("No fruit detected.")

        detection_image = (
            classification_image.copy()
        )

        fruit_crops = []

    else:

        print(
            f"Total fruits detected: "
            f"{len(final_detections)}"
        )

        for index, detection in enumerate(
            final_detections,
            start=1
        ):

            print(
                f"\nFruit {index}"
            )

            print(
                f"Fruit      : "
                f"{detection['fruit_type']}"
            )

            print(
                f"Confidence : "
                f"{detection['confidence'] * 100:.2f}%"
            )

            print(
                f"Bounding box : "
                f"{detection['bounding_box']}"
            )

            print(
                f"Agreement   : "
                f"{detection['agreement']}"
            )

            if "iou" in detection:

                print(
                    f"Model IoU   : "
                    f"{detection['iou']:.2f}"
                )

        # ====================================================
        # DRAW ALL FRUIT BOXES
        # ====================================================

        detection_image = (
            draw_final_detections(
                classification_image,
                final_detections
            )
        )

        # ====================================================
        # CROP ALL FRUITS
        # ====================================================

        fruit_crops = (
            crop_all_detected_fruits(
                classification_image,
                final_detections,
                margin_ratio=0.10
            )
        )
    
    # ========================================================
    # FRUIT DETECTION
    # ========================================================
    display_detection_image = resize_for_display(
        detection_image
    )

    cv2.namedWindow(
        "Fruit Detection",
        cv2.WINDOW_NORMAL
    )

    detection_height, detection_width = (
        display_detection_image.shape[:2]
    )

    cv2.resizeWindow(
        "Fruit Detection",
        detection_width,
        detection_height
    )

    cv2.imshow(
        "Fruit Detection",
        display_detection_image
    )


    # ========================================================
    # SHOW ALL DETECTED FRUIT CROPS
    # ========================================================
    for fruit_crop_result in fruit_crops:

        fruit_index = (
            fruit_crop_result["index"]
        )

        fruit_type = (
            fruit_crop_result[
                "fruit_type"
            ]
        )

        crop_image = (
            fruit_crop_result[
                "crop"
            ]
        )

        window_name = (
            f"Fruit {fruit_index} - "
            f"{fruit_type}"
        )

        cv2.namedWindow(
            window_name,
            cv2.WINDOW_NORMAL
        )

        cv2.imshow(
            window_name,
            crop_image
        )


    print(
        "\nPress any key on the Fruit Detection window to continue."
    )

    cv2.waitKey(0)
    cv2.destroyAllWindows()

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
    # TECHNIQUE 5: SEPARATE MORPHOLOGICAL REFINEMENT
    # ========================================================

    # Refine grayscale Otsu mask separately
    (
        gray_opened_mask,
        gray_refined_mask
    ) = refine_fruit_mask(
        gray_mask,
        opening_kernel_size=3,
        closing_kernel_size=5
    )


    # Refine saturation Otsu mask separately
    (
        saturation_opened_mask,
        saturation_refined_mask
    ) = refine_fruit_mask(
        saturation_mask,
        opening_kernel_size=3,
        closing_kernel_size=5
    )


    # ========================================================
    # TECHNIQUE 5.1: COMBINE REFINED OTSU MASKS
    # ========================================================

    combined_mask = combine_otsu_masks_constrained(
        gray_refined_mask,
        saturation_refined_mask,
        expansion_kernel_size=9
    )


    # Final light refinement after combination
    (
        opened_mask,
        refined_mask
    ) = refine_fruit_mask(
        combined_mask,
        opening_kernel_size=3,
        closing_kernel_size=5
    )

    print("\nMorphological Refinement")
    print("------------------------------")

    print(
        "Grayscale mask refined separately."
    )

    print(
        "Saturation mask refined separately."
    )

    print(
        "Refined masks combined using "
        "constrained mask combination."
    )

    print(
        "Final combined mask refined."
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
    # ========================================================
    # TECHNIQUE 8: Colour preserve and feature extraction
    # ========================================================

    fruit_only_colour = cv2.bitwise_and(
        working_image,
        working_image,
        mask=fruit_mask
    )

    colour_features = extract_colour_features(
        working_image,
        fruit_mask
    )

    print("\nFruit Colour Features")
    print("------------------------------")

    print(
        f"Mean Red        : "
        f"{colour_features['mean_red']:.2f}"
    )

    print(
        f"Mean Green      : "
        f"{colour_features['mean_green']:.2f}"
    )

    print(
        f"Mean Blue       : "
        f"{colour_features['mean_blue']:.2f}"
    )

    print(
        f"Dominant Hue    : "
        f"{colour_features['dominant_hue']}"
    )

    print(
        f"Mean Saturation : "
        f"{colour_features['mean_saturation']:.2f}"
    )

    print(
        f"Mean Value      : "
        f"{colour_features['mean_value']:.2f}"
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
    # MEMBER 2: ROI-BASED SEGMENTATION
    # ========================================================
    roi_results = []

    print("\nROI Fruit Processing")
    print("------------------------------")

    if len(final_detections) == 0:

        print(
            "Skipped - no fruit detections available."
        )

    else:

        for index, detection in enumerate(
            final_detections,
            start=1
        ):

            try:

                roi_result = process_fruit_roi(
                    working_image,
                    detection[
                        "bounding_box"
                    ],
                    use_watershed=False
                )

                # ============================================
                # ATTACH FRUIT DETECTION INFORMATION
                # ============================================

                roi_result[
                    "fruit_index"
                ] = index

                roi_result[
                    "fruit_type"
                ] = detection[
                    "fruit_type"
                ]

                roi_result[
                    "detection_model"
                ] = detection[
                    "agreement"
                ]

                roi_result[
                    "detection_confidence"
                ] = detection[
                    "confidence"
                ]

                # Save Model C ripeness for later.
                # Do not use as final ripeness yet.
                roi_result[
                    "model_c_ripeness"
                ] = detection.get(
                    "model_c_ripeness"
                )

                roi_results.append(
                    roi_result
                )

                roi_colour_features = (
                    roi_result[
                        "colour_features"
                    ]
                )

                print(
                    f"\nFruit ROI {index}"
                )

                print(
                    f"Fruit        : "
                    f"{roi_result['fruit_type']}"
                )

                print(
                    f"Detection    : "
                    f"{roi_result['detection_model']}"
                )

                print(
                    f"Confidence   : "
                    f"{roi_result['detection_confidence'] * 100:.2f}%"
                )

                print(
                    f"Bounding box : "
                    f"{roi_result['bounding_box']}"
                )

                print(
                    f"ROI area     : "
                    f"{roi_result['fruit_area_pixels']} "
                    f"pixels^2"
                )

                print(
                    f"Mean Red     : "
                    f"{roi_colour_features['mean_red']:.2f}"
                )

                print(
                    f"Mean Green   : "
                    f"{roi_colour_features['mean_green']:.2f}"
                )

                print(
                    f"Mean Blue    : "
                    f"{roi_colour_features['mean_blue']:.2f}"
                )

                print(
                    f"Dominant Hue : "
                    f"{roi_colour_features['dominant_hue']}"
                )

                print(
                    f"Mean Saturation : "
                    f"{roi_colour_features['mean_saturation']:.2f}"
                )

                print(
                    f"Mean Value      : "
                    f"{roi_colour_features['mean_value']:.2f}"
                )

            except ValueError as error:

                print(
                    f"\nFruit ROI {index} failed:"
                )

                print(error)


    # ========================================================
    # MEMBER 3: RIPENESS CLASSIFICATION
    # ========================================================
    print("\nRipeness Classification\n------------------------------")

    ripeness_results = []

    # Image for final ripeness results
    ripeness_image = working_image.copy()

    if len(roi_results) == 0:
        print(
            "Skipped - no segmented fruit ROIs available."
        )

    else:
        for index, roi_result in enumerate(
            roi_results,
            start=1
        ):

            fruit_type = roi_result[
                "fruit_type"
            ]

            bounding_box = roi_result[
                "bounding_box"
            ]

            x1, y1, x2, y2 = (
                bounding_box
            )

            # =================================================
            # GET FRUIT ROI
            # =================================================
            fruit_roi = working_image[
                y1:y2,
                x1:x2
            ].copy()

            if fruit_roi.size == 0:

                print(
                    f"Fruit {index}: "
                    f"invalid ROI."
                )

                continue

            # =================================================
            # MODEL B
            # =================================================
            result_b = (
                classify_with_model_b(
                    fruit_roi,
                    fruit_type
                )
            )

            # =================================================
            # MODEL E
            # =================================================
            result_e = (
                classify_with_model_e(
                    fruit_roi
                )
            )

            # =================================================
            # MODEL C RESULT FROM DETECTION STAGE
            # =================================================
            model_c_ripeness = (
                roi_result.get(
                    "model_c_ripeness"
                )
            )

            # Find corresponding final detection
            detection_index = (
                roi_result[
                    "fruit_index"
                ]
                - 1
            )

            final_detection = (
                final_detections[
                    detection_index
                ]
            )

            model_c_confidence = (
                final_detection.get(
                    "confidence_c"
                )
            )

            # =================================================
            # FUSE B + C + E
            # =================================================
            final_ripeness = (
                fuse_ripeness(
                    result_b,
                    model_c_ripeness,
                    model_c_confidence,
                    result_e
                )
            )

            # =================================================
            # SAVE RESULT
            # =================================================
            ripeness_result = {
                "fruit_index": index,

                "fruit_type": fruit_type,

                "bounding_box": (
                    bounding_box
                ),

                "model_b_ripeness": (
                    result_b[
                        "ripeness"
                    ]
                ),

                "model_b_confidence": (
                    result_b[
                        "confidence"
                    ]
                ),

                "model_c_ripeness": (
                    model_c_ripeness
                ),

                "model_c_confidence": (
                    model_c_confidence
                ),

                "model_e_ripeness": (
                    result_e[
                        "ripeness"
                    ]
                ),

                "model_e_confidence": (
                    result_e[
                        "confidence"
                    ]
                ),

                "final_ripeness": (
                    final_ripeness[
                        "ripeness"
                    ]
                ),

                "final_confidence": (
                    final_ripeness[
                        "confidence"
                    ]
                ),

                "fusion_scores": (
                    final_ripeness[
                        "scores"
                    ]
                )
            }

            ripeness_results.append(
                ripeness_result
            )

            # Also attach final result back to ROI result.
            roi_result[
                "ripeness"
            ] = final_ripeness[
                "ripeness"
            ]

            roi_result[
                "ripeness_confidence"
            ] = final_ripeness[
                "confidence"
            ]

            # =================================================
            # PRINT RESULT
            # =================================================
            print(
                f"Fruit {index}"
            )

            print(
                f"Fruit Type : "
                f"{fruit_type}"
            )

            if result_b.get(
                "available",
                False
            ):

                print(
                    f"Model B    : "
                    f"{result_b['ripeness']} "
                    f"({result_b['confidence'] * 100:.2f}%)"
                )

            else:
            
                print(
                    "Model B    : "
                    "Not available for this fruit"
                )

            if model_c_ripeness is not None:
                if model_c_confidence is not None:
                    print(
                        f"Model C    : "
                        f"{model_c_ripeness} "
                        f"({model_c_confidence * 100:.2f}%)"
                    )

                else:
                    print(
                        f"Model C    : "
                        f"{model_c_ripeness}"
                    )

            else:
                print(
                    "Model C    : "
                    "Not available for this fruit"
                )

            print(
                f"Model E    : "
                f"{result_e['ripeness']} "
                f"({result_e['confidence'] * 100:.2f}%)"
            )

            print(
                f"Final      : "
                f"{final_ripeness['ripeness']} "
                f"({final_ripeness['confidence'] * 100:.2f}%)\n"
            )

            # =================================================
            # DRAW FINAL RIPENESS RESULT
            # =================================================

            final_label = (
                f"{final_ripeness['ripeness']} "
                f"{final_ripeness['confidence'] * 100:.2f}%"
            )

            # Draw fruit bounding box
            cv2.rectangle(
                ripeness_image,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                3
            )

            # Get text size
            (text_width, text_height), baseline = (
                cv2.getTextSize(
                    final_label,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    2
                )
            )

            # Position label above bounding box
            text_x = x1
            text_y = max(
                y1 - 10,
                text_height + 10
            )

            # Draw background behind text
            cv2.rectangle(
                ripeness_image,
                (
                    text_x,
                    text_y - text_height - 8
                ),
                (
                    text_x + text_width + 8,
                    text_y + baseline
                ),
                (0, 0, 255),
                -1
            )

            # Draw final ripeness text
            cv2.putText(
                ripeness_image,
                final_label,
                (
                    text_x + 4,
                    text_y - 4
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
                cv2.LINE_AA
            )

#    # ========================================================
#    # DISPLAY FINAL RIPENESS IMAGE
#    # ========================================================
#
#    if len(ripeness_results) > 0:
#
#        display_ripeness_image = resize_for_display(
#            ripeness_image
#        )
#
#        cv2.namedWindow(
#            "Final Ripeness Classification",
#            cv2.WINDOW_NORMAL
#        )
#
#        ripeness_height, ripeness_width = (
#            display_ripeness_image.shape[:2]
#        )
#
#        cv2.resizeWindow(
#            "Final Ripeness Classification",
#            ripeness_width,
#            ripeness_height
#        )
#
#        cv2.imshow(
#            "Final Ripeness Classification",
#            display_ripeness_image
#        )
#
#        print(
#            "\nPress any key on the Final Ripeness "
#            "Classification window to continue."
#        )
#
#        cv2.waitKey(0)
#        cv2.destroyAllWindows()

#    # ========================================================
#    # MEMBER 3: BLEMISH ANALYSIS
#    # ========================================================
#    blemish_results = None
#
#    if len(detections) == 0:
#
#        print("\nBlemish Analysis")
#        print("------------------------------")
#        print("Skipped - no detected fruit class available.")
#
#    else:
#        # Use the highest-confidence detection
#        primary_detection = max(
#            detections,
#            key=lambda detection: detection["confidence"]
#        )
#
#        primary_fruit_type = primary_detection[
#            "fruit_type"
#        ]
#
#        try:
#            blemish_results = detect_fruit_blemish(
#                image=working_image,
#                fruit_mask=fruit_mask,
#                fruit_type=primary_fruit_type,
#                opening_kernel_size=3,
#                closing_kernel_size=5,
#                min_component_area=60
#            )
#
#        except (TypeError, ValueError, cv2.error) as error:
#            print(
#                "\nBlemish Analysis"
#            )
#            print(
#                "------------------------------"
#            )
#            print(
#                "Skipped - blemish analysis failed:"
#            )
#            print(error)
#
#            blemish_results = None
#
#    if blemish_results is not None:
#        
#        print("\nBlemish Analysis")
#        print("------------------------------")
#        print(
#            f"Fruit type used    : "
#            f"{blemish_results['fruit_type_used']}"
#        )
#        print(
#            f"Blemish area       : "
#            f"{blemish_results['blemish_area_pixels']} pixels^2"
#        )
#        print(
#            f"Blemish Percentage : "
#            f"{blemish_results['blemish_percentage']:.2f}%"
#        )

    # ========================================================
    # RETURN RESULTS
    # ========================================================

    return {
    "analysis_image": analysis_image,
    "classification_image": classification_image,
    "working_image": working_image,

    "roi_results": roi_results,

    "gray_image": gray_image,
    "gray_mask": gray_mask,
    "gray_opened_mask": gray_opened_mask,
    "gray_refined_mask": gray_refined_mask,

    "saturation_image": saturation_image,
    "saturation_mask": saturation_mask,
    "saturation_opened_mask": saturation_opened_mask,
    "saturation_refined_mask": saturation_refined_mask,

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

    "fruit_only_colour": fruit_only_colour,
    "colour_features": colour_features,

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

    "detections_a": detections_a,
    "detections_c": detections_c,
    "detections_d": detections_d,

    "final_detections": final_detections,

    "fruit_crops": fruit_crops,

    "detection_image": detection_image,

    "ripeness_results": ripeness_results,
    "ripeness_image": ripeness_image,

#    "blemish_mask": (
#        blemish_results["blemish_mask"]
#        if blemish_results is not None else None
#    ),
#
#    "blemish_overlay": (
#        blemish_results["blemish_overlay"]
#        if blemish_results is not None else None
#    ),
#
#    "blemish_area_pixels": (
#        blemish_results["blemish_area_pixels"]
#        if blemish_results is not None else 0
#    ),
#
#    "blemish_percentage": (
#        blemish_results["blemish_percentage"]
#        if blemish_results is not None else 0.0
#    ),
}


if __name__ == "__main__":
    # Select Image
    root = Tk()
    root.withdraw()

    try:
        image_path = filedialog.askopenfilename(
            title="Select Fruit Image",
            filetypes=[
                ("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp"),
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

    # ========================================================
    # DISPLAY MEMBER 2 RESULTS
    # ========================================================
    
    display_results(results)
    
    display_roi_results(
        results["roi_results"]
    )
    
    # Save results after displaying
    save_results(results)
    
    print(
        "\nPress any key on the image windows to continue to Ripeness Classification."
    )
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
    # ========================================================
    # DISPLAY MEMBER 3 RIPENESS RESULT
    # ========================================================
    
    if len(results["ripeness_results"]) > 0:
    
        display_ripeness_image = resize_for_display(
            results["ripeness_image"]
        )
    
        cv2.namedWindow(
            "Final Ripeness Classification",
            cv2.WINDOW_NORMAL
        )
    
        ripeness_height, ripeness_width = (
            display_ripeness_image.shape[:2]
        )
    
        cv2.resizeWindow(
            "Final Ripeness Classification",
            ripeness_width,
            ripeness_height
        )
    
        cv2.imshow(
            "Final Ripeness Classification",
            display_ripeness_image
        )
    
        print(
            "\nPress any key on the Final Ripeness Classification window to end the program."
        )
    
        cv2.waitKey(0)
        cv2.destroyAllWindows()



#    # Blemish calculation
#    if results["blemish_mask"] is not None:
#        cv2.namedWindow(
#            "Blemish Detection",
#            cv2.WINDOW_NORMAL
#        )
#
#        cv2.imshow(
#            "Blemish Detection",
#            results["blemish_mask"]
#        )
#
#        cv2.namedWindow(
#            "Blemish Overlay",
#            cv2.WINDOW_NORMAL
#        )
#
#        cv2.imshow(
#            "Blemish Overlay",
#            results["blemish_overlay"]
#        )
#
#        print(
#            f"\nBlemish Percentage: "
#            f"{results['blemish_percentage']:.2f}%"
#        )
#
#        print("\nPress any key on an image window to exit the program.")
#
#        cv2.waitKey(0)
#        cv2.destroyAllWindows()
        