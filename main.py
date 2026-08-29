import traceback
import cv2
from tkinter import Tk, filedialog

from preprocessing import (
    preprocess_fruit_image
)

from calibration_segmentation.roi_processing import (
    process_fruit_roi
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
    extract_main_fruit
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
    assess_detection_quality,
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


def boxes_represent_same_region(
    first_box,
    second_box,
    minimum_iou=0.25,
    minimum_smaller_box_coverage=0.60
):
    """Check whether two boxes most likely describe the same fruit region."""
    first_x1, first_y1, first_x2, first_y2 = first_box
    second_x1, second_y1, second_x2, second_y2 = second_box

    intersection_width = max(
        0,
        min(first_x2, second_x2)
        - max(first_x1, second_x1)
    )
    intersection_height = max(
        0,
        min(first_y2, second_y2)
        - max(first_y1, second_y1)
    )
    intersection_area = intersection_width * intersection_height
    first_area = max(0, first_x2 - first_x1) * max(
        0,
        first_y2 - first_y1
    )
    second_area = max(0, second_x2 - second_x1) * max(
        0,
        second_y2 - second_y1
    )
    union_area = first_area + second_area - intersection_area
    smaller_area = min(first_area, second_area)

    iou = (
        intersection_area / union_area
        if union_area > 0
        else 0.0
    )
    smaller_box_coverage = (
        intersection_area / smaller_area
        if smaller_area > 0
        else 0.0
    )

    first_width = max(1, first_x2 - first_x1)
    first_height = max(1, first_y2 - first_y1)
    second_width = max(1, second_x2 - second_x1)
    second_height = max(1, second_y2 - second_y1)
    first_center_x = (first_x1 + first_x2) / 2.0
    first_center_y = (first_y1 + first_y2) / 2.0
    second_center_x = (second_x1 + second_x2) / 2.0
    second_center_y = (second_y1 + second_y2) / 2.0
    horizontal_center_distance = abs(
        first_center_x - second_center_x
    ) / min(first_width, second_width)
    vertical_center_distance = abs(
        first_center_y - second_center_y
    ) / min(first_height, second_height)
    centers_are_aligned = (
        horizontal_center_distance <= 0.35
        and vertical_center_distance <= 0.35
    )

    return (
        centers_are_aligned
        and (
            iou >= minimum_iou
            or smaller_box_coverage
            >= minimum_smaller_box_coverage
        )
    )


def draw_ripeness_results(
    image,
    ripeness_results
):
    """Draw final ripeness result for evaluated fruits."""

    output_image = image.copy()

    for result in ripeness_results:

        if (
            result.get("final_ripeness") is None
            or result.get("final_confidence") is None
        ):
            continue

        x1, y1, x2, y2 = result["bounding_box"]

        final_label = (
            f"{result['final_ripeness']} "
            f"{result['final_confidence'] * 100:.2f}%"
        )

        cv2.rectangle(
            output_image,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            3
        )

        (text_width, text_height), baseline = (
            cv2.getTextSize(
                final_label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                2
            )
        )

        text_x = x1
        text_y = max(
            y1 - 10,
            text_height + 10
        )

        cv2.rectangle(
            output_image,
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

        cv2.putText(
            output_image,
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

    return output_image

#def draw_ripeness_results(
#    image,
#    ripeness_results,
#    detection_only_boxes=None
#):
#    """Draw boxes only for fruits that received a ripeness result."""
#    output_image = image.copy()
#    excluded_boxes = detection_only_boxes or []
#
#    for result in ripeness_results:
#        if (
#            not result.get("ripeness_supported", False)
#            or result.get("final_ripeness") is None
#            or result.get("final_confidence") is None
#        ):
#            continue
#
#        if any(
#            boxes_represent_same_region(
#                result["bounding_box"],
#                excluded_box
#            )
#            for excluded_box in excluded_boxes
#        ):
#            continue
#
#        x1, y1, x2, y2 = result["bounding_box"]
#        final_label = (
#            f"{result['final_ripeness']} "
#            f"{result['final_confidence'] * 100:.2f}%"
#        )
#
#        cv2.rectangle(
#            output_image,
#            (x1, y1),
#            (x2, y2),
#            (0, 0, 255),
#            3
#        )
#
#        (text_width, text_height), baseline = cv2.getTextSize(
#            final_label,
#            cv2.FONT_HERSHEY_SIMPLEX,
#            0.6,
#            2
#        )
#        text_x = x1
#        text_y = max(y1 - 10, text_height + 10)
#
#        cv2.rectangle(
#            output_image,
#            (text_x, text_y - text_height - 8),
#            (text_x + text_width + 8, text_y + baseline),
#            (0, 0, 255),
#            -1
#        )
#
#        cv2.putText(
#            output_image,
#            final_label,
#            (text_x + 4, text_y - 4),
#            cv2.FONT_HERSHEY_SIMPLEX,
#            0.6,
#            (0, 0, 0),
#            2,
#            cv2.LINE_AA
#        )
#
#    return output_image

def run_fruit_assessment(
    image_path,
    use_watershed=True
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

    valid_content_bbox = preprocessing_results[
        "valid_content_bbox"
    ]

    preprocessing_suitability = preprocessing_results[
        "preprocessing_suitability"
    ]

    blur_status = preprocessing_results[
        "blur_status"
    ]

    exposure_status = preprocessing_results[
        "exposure_status"
    ]

    contrast_status = preprocessing_results[
        "contrast_status"
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
        f"Preprocessing suitability: "
        f"{preprocessing_suitability}"
    )

    print(
        f"Blur: {blur_status}"
    )

    print(
        f"Exposure: {exposure_status}"
    )

    print(
        f"Contrast: {contrast_status}"
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

    final_detections = assess_detection_quality(
        final_detections,
        classification_image.shape,
        valid_content_bbox=valid_content_bbox,
        retain_rejected=True,
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

            print(
                f"Reliability : "
                f"{detection['reliability_status']}"
            )

            print(
                f"Box quality : "
                f"{detection['box_status']}"
            )

            print(
                f"Support     : "
                f"{detection['support_level']}"
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

    # Use the classifier-ready image throughout downstream processing so
    # detected coordinates and classifier crops refer to the same input.
    working_image = classification_image.copy()


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
            foreground_ratio=0.55
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

    print(
        "Physical projected area : "
        "Not available without spatial calibration"
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

            if detection.get("reliability_status") == "Rejected":
                print(
                    f"\nFruit ROI {index} skipped: "
                    + "; ".join(
                        detection.get(
                            "reliability_reasons",
                            ["Detection was rejected"],
                        )
                    )
                )
                continue

            try:

                roi_result = process_fruit_roi(
                    working_image,
                    detection[
                        "bounding_box"
                    ],
                    use_watershed=use_watershed,
                    global_refined_mask=refined_mask
                )

                print(
                    f"Bounding box : "
                    f"{roi_result['bounding_box']}"
                )

                print(
                    f"Watershed    : "
                    f"{'Enabled' if roi_result.get('watershed_used', False) else 'Disabled'}"
                )

                if roi_result.get(
                    "fruit_labels"
                ) is not None:

                    print(
                        f"Regions      : "
                        f"{len(roi_result['fruit_labels'])}"
                    )

                print(
                    f"ROI area     : "
                    f"{roi_result['fruit_area_pixels']} "
                    f"pixels^2"
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

                roi_result[
                    "detection_reliability"
                ] = detection.get(
                    "reliability_status"
                )

                roi_result[
                    "ripeness_supported"
                ] = detection.get(
                    "ripeness_supported",
                    False
                )

                roi_result[
                    "defect_supported"
                ] = detection.get(
                    "defect_supported",
                    False
                )

                BLEMISH_SUPPORTED_FRUITS = {
                    "apple",
                    "banana",
                    "grape",
                    "mango",
                    "melon",
                    "orange",
                    "peach",
                    "pear",
                    "pineapple",
                    "watermelon"
                }

                roi_result[
                    "blemish_supported"
                ] = (
                    str(detection["fruit_type"]).strip().lower()
                    in BLEMISH_SUPPORTED_FRUITS
                )

                roi_result[
                    "support_level"
                ] = detection.get(
                    "support_level",
                    "Detection only"
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
    ripeness_image = fruit_only_colour.copy()

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

            if not roi_result.get(
                "ripeness_supported",
                False
            ):
                roi_result["ripeness"] = None
                roi_result["ripeness_confidence"] = None
                roi_result["ripeness_status"] = (
                    "Not evaluated - detection only"
                )

                print(
                    f"Fruit {index}"
                )

                print(
                    f"Fruit Type : "
                    f"{fruit_type}"
                )

                print(
                    "Ripeness   : "
                    "Not evaluated - detection only\n"
                )

                continue

            #if not roi_result.get(
            #    "ripeness_supported",
            #    False
            #):
            #    roi_result["ripeness"] = None
            #    roi_result["ripeness_confidence"] = None
            #    roi_result["ripeness_status"] = (
            #        "Not evaluated - detection only"
            #    )
            #    continue

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

                "ripeness_supported": True,

                "evaluation_status": "Evaluated",

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

    # Build the ripeness image exclusively from evaluated ripeness results.
    # Detection-only fruits therefore cannot receive a ripeness boundary.
    ripeness_image = draw_ripeness_results(
        fruit_only_colour,
        ripeness_results,
        #detection_only_boxes=[
        #    detection["bounding_box"]
        #    for detection in final_detections
        #    if not detection.get(
        #        "ripeness_supported",
        #        False
        #    )
        #]
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

    # ========================================================
    # MEMBER 3: BLEMISH ANALYSIS
    # ========================================================

    print("\nBlemish Analysis")
    print("------------------------------")

    blemish_results = []

    if len(roi_results) == 0:
        print(
            "Skipped - no segmented fruit ROIs available."
        )

    else:
        for roi_result in roi_results:
            fruit_index = roi_result[
                "fruit_index"
            ]

            fruit_type = roi_result[
                "fruit_type"
            ]

            # Only analyse fruits supported for defect detection
            if not roi_result.get(
                "blemish_supported",
                False
            ):

                print(
                    f"Fruit {fruit_index} - "
                    f"{fruit_type}: "
                    f"Blemish analysis not supported."
                )

                continue

            try:
                # Member 2 ROI image
                roi_image = roi_result[
                    "roi_image"
                ]

                # Member 2 segmented fruit mask
                roi_fruit_mask = roi_result[
                    "fruit_mask"
                ]

                blemish_result = (
                    detect_fruit_blemish(
                        image=roi_image,
                        fruit_mask=roi_fruit_mask,
                        fruit_type=fruit_type,
                        opening_kernel_size=3,
                        closing_kernel_size=5,
                        min_component_area=60
                    )
                )

                #print("\nBlemish Debug")
                #print("------------------------------")
                #print("Fruit index :", fruit_index)
                #print("Fruit type  :", fruit_type)
                #
                #print(
                #    "ROI image   :",
                #    None if roi_result.get("roi_image") is None
                #    else roi_result["roi_image"].shape
                #)
                #
                #print(
                #    "Fruit mask  :",
                #    None if roi_result.get("fruit_mask") is None
                #    else roi_result["fruit_mask"].shape
                #)
                #
                #print(
                #    "Fruit area  :",
                #    roi_result.get("fruit_area_pixels")
                #)

                # Add fruit information
                blemish_result[
                    "fruit_index"
                ] = fruit_index

                blemish_result[
                    "fruit_type"
                ] = fruit_type

                blemish_result[
                    "bounding_box"
                ] = roi_result[
                    "bounding_box"
                ]

                blemish_results.append(
                    blemish_result
                )

                # Attach back to ROI result
                roi_result[
                    "blemish_percentage"
                ] = blemish_result[
                    "blemish_percentage"
                ]

                roi_result[
                    "blemish_area_pixels"
                ] = blemish_result[
                    "blemish_area_pixels"
                ]

                print(
                    f"Fruit {fruit_index}"
                )

                print(
                    f"Fruit Type         : "
                    f"{fruit_type}"
                )

                print(
                    f"Fruit area         : "
                    f"{blemish_result['fruit_area_pixels']} "
                    f"pixels^2"
                )

                print(
                    f"Blemish area       : "
                    f"{blemish_result['blemish_area_pixels']} "
                    f"pixels^2"
                )

                print(
                    f"Blemish Percentage : "
                    f"{blemish_result['blemish_percentage']:.2f}%\n"
                )

            except (
                KeyError,
                TypeError,
                ValueError,
                cv2.error
            ) as error:

                print(
                    f"\nFruit {fruit_index} "
                    f"blemish analysis failed:"
                )

                print(error)

                traceback.print_exc()

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
    "blur_score": blur_score,
    "is_blurry": is_blurry,

    "preprocessing_suitability": preprocessing_suitability,
    "blur_status": blur_status,
    "exposure_status": exposure_status,
    "contrast_status": contrast_status,

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

    "blemish_results": blemish_results,

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
        use_watershed=False
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
            "\nPress any key on the Final Ripeness Classification window to continue."
        )
    
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    # ========================================================
    # DISPLAY MEMBER 3 BLEMISH RESULTS
    # ========================================================
    
    for blemish_result in results[
        "blemish_results"
    ]:
    
        fruit_index = blemish_result[
            "fruit_index"
        ]
    
        fruit_type = blemish_result[
            "fruit_type"
        ]
    
        blemish_percentage = blemish_result[
            "blemish_percentage"
        ]
    
        overlay = blemish_result[
            "blemish_overlay"
        ].copy()
    
        label = (
            f"Blemish {blemish_percentage:.2f}%"
        )

        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            2
        )

        text_x = 10
        text_y = text_height + 12

        # background rectangle
        cv2.rectangle(
            overlay,
            (text_x, text_y - text_height - 8),
            (text_x + text_width + 10, text_y + baseline),
            (255, 0, 255),   # purple background
            -1
        )

        # black text
        cv2.putText(
            overlay,
            label,
            (text_x + 5, text_y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),       # black text
            2,
            cv2.LINE_AA
        )
    
        window_name = (
            f"Fruit {fruit_index} - "
            f"{fruit_type} Blemish"
        )
    
        cv2.namedWindow(
            window_name,
            cv2.WINDOW_NORMAL
        )
    
        cv2.imshow(
            window_name,
            overlay
        )
    
    if len(results["blemish_results"]) > 0:
    
        print(
            "\nPress any key on the Blemish Analysis "
            "windows to end the program."
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
