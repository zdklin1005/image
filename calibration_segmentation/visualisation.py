import cv2
import os
import numpy as np

def display_roi_results(
    roi_results
):
    """
    Display one combined panel for each ROI.
    """

    if not roi_results:
        print(
            "No ROI results available for visualisation."
        )
        return

    for index, roi_result in enumerate(
        roi_results,
        start=1
    ):

        panel = create_roi_visualisation_panel(
            roi_result,
            index
        )

        window_name = (
            f"ROI {index} - Processing Results"
        )

        cv2.namedWindow(
            window_name,
            cv2.WINDOW_NORMAL
        )

        cv2.imshow(
            window_name,
            panel
        )

        cv2.resizeWindow(
            window_name,
            panel.shape[1],
            panel.shape[0]
        )

def prepare_panel_image(image, size=(420, 300)):
    """
    Resize an image for panel display.
    Converts grayscale images to BGR so all
    panel images have the same number of channels.
    """

    if image is None:
        return None

    if len(image.shape) == 2:
        image = cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2BGR
        )

    return cv2.resize(
        image,
        size,
        interpolation=cv2.INTER_AREA
    )

def add_panel_label(image, label):
    """
    Add a simple label at the top-left of a panel image.
    """

    labelled_image = image.copy()

    cv2.putText(
        labelled_image,
        label,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        labelled_image,
        label,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        1,
        cv2.LINE_AA
    )

    return labelled_image

def create_roi_visualisation_panel(
    roi_result,
    index
):
    """
    Create one combined visualisation panel
    for a YOLO-detected fruit ROI.
    """

    fruit_type = roi_result.get(
        "fruit_type",
        "Fruit"
    )

    ripeness = roi_result.get(
        "ripeness",
        "Unknown"
    )

    confidence = roi_result.get(
        "confidence",
        0.0
    )

    # ----------------------------------------------------
    # Prepare images
    # ----------------------------------------------------

    roi_image = prepare_panel_image(
        roi_result["roi_image"]
    )

    gray_mask = prepare_panel_image(
        roi_result["gray_refined_mask"]
    )

    saturation_mask = prepare_panel_image(
        roi_result["saturation_refined_mask"]
    )

    combined_mask = prepare_panel_image(
        roi_result["combined_mask"]
    )

    final_mask = prepare_panel_image(
        roi_result["fruit_mask"]
    )

    fruit_colour = prepare_panel_image(
        roi_result["fruit_only_colour"]
    )

    # ----------------------------------------------------
    # Add labels
    # ----------------------------------------------------

    roi_image = add_panel_label(
        roi_image,
        "ROI Image"
    )

    gray_mask = add_panel_label(
        gray_mask,
        "Grayscale Mask"
    )

    saturation_mask = add_panel_label(
        saturation_mask,
        "Saturation Mask"
    )

    combined_mask = add_panel_label(
        combined_mask,
        "Combined Mask"
    )

    final_mask = add_panel_label(
        final_mask,
        "Final Fruit Mask"
    )

    fruit_colour = add_panel_label(
        fruit_colour,
        "Fruit Only Colour"
    )

    # ----------------------------------------------------
    # Build 2 x 3 panel
    # ----------------------------------------------------

    top_row = cv2.hconcat([
        roi_image,
        gray_mask,
        saturation_mask
    ])

    bottom_row = cv2.hconcat([
        combined_mask,
        final_mask,
        fruit_colour
    ])

    panel = cv2.vconcat([
        top_row,
        bottom_row
    ])

    # ----------------------------------------------------
    # Add header
    # ----------------------------------------------------

    header_height = 55

    header = np.zeros(
        (
            header_height,
            panel.shape[1],
            3
        ),
        dtype=np.uint8
    )

    header_text = (
        f"ROI {index} | "
        f"{fruit_type} | "
        f"{ripeness} | "
        f"Confidence: {confidence * 100:.2f}%"
    )

    cv2.putText(
        header,
        header_text,
        (15, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    panel = cv2.vconcat([
        header,
        panel
    ])

    return panel

def create_contour_image(image, fruit_contour):
    """
    Create a copy of the image with the detected
    fruit contour drawn on top.
    """

    if image is None:
        raise ValueError(
            "Input image cannot be None."
        )

    if fruit_contour is None:
        raise ValueError(
            "Fruit contour cannot be None."
        )

    contour_image = image.copy()

    cv2.drawContours(
        contour_image,
        [fruit_contour],
        -1,
        (0, 0, 255),
        2
    )

    return contour_image

def create_watershed_visualisation(
    image,
    watershed_markers
):
    """
    Draw Watershed boundaries on a copy of the colour image.

    Watershed boundaries are represented by marker value -1.
    """

    if image is None:
        raise ValueError(
            "Input image cannot be None."
        )

    if watershed_markers is None:
        raise ValueError(
            "Watershed markers cannot be None."
        )

    if image.shape[:2] != watershed_markers.shape[:2]:
        raise ValueError(
            "Image and Watershed marker dimensions must match."
        )

    watershed_image = image.copy()

    # Mark Watershed boundaries in red
    watershed_image[
        watershed_markers == -1
    ] = [0, 0, 255]

    return watershed_image

def display_results(results):
    """
    Display the important processing results.
    """

    working_image = results[
        "working_image"
    ]

    saturation_mask = results[
        "saturation_mask"
    ]

    refined_mask = results[
        "refined_mask"
    ]

    fruit_mask = results[
        "fruit_mask"
    ]

    fruit_contour = results[
        "fruit_contour"
    ]

    fruit_only_colour = results[
        "fruit_only_colour"
    ]

    contour_image = create_contour_image(
        working_image,
        fruit_contour
    )




    cv2.namedWindow(
        "Working Image",
        cv2.WINDOW_NORMAL
    )

    cv2.namedWindow(
        "Saturation Otsu Mask",
        cv2.WINDOW_NORMAL
    )

    cv2.namedWindow(
        "Refined Fruit Mask",
        cv2.WINDOW_NORMAL
    )

    cv2.namedWindow(
        "Final Fruit Mask",
        cv2.WINDOW_NORMAL
    )

    cv2.namedWindow(
        "Fruit Only Colour",
        cv2.WINDOW_NORMAL
    )

    cv2.namedWindow(
        "Fruit Contour",
        cv2.WINDOW_NORMAL
    )

    cv2.imshow(
        "Working Image",
        working_image
    )

    cv2.imshow(
        "Saturation Otsu Mask",
        saturation_mask
    )

    cv2.imshow(
        "Refined Fruit Mask",
        refined_mask
    )

    cv2.imshow(
        "Final Fruit Mask",
        fruit_mask
    )

    cv2.imshow(
        "Fruit Contour",
        contour_image
    )

    cv2.imshow(
        "Fruit Only Colour",
        fruit_only_colour
    )
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()


def save_results(
    results,
    output_dir="calibration_segmentation/outputs"
):
    """
    Save important processing results to disk.
    """

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    working_image = results[
        "working_image"
    ]

    gray_mask = results[
        "gray_mask"
    ]

    combined_mask = results[
        "combined_mask"
    ]

    saturation_mask = results[
        "saturation_mask"
    ]

    refined_mask = results[
        "refined_mask"
    ]

    fruit_mask = results[
        "fruit_mask"
    ]

    fruit_contour = results[
        "fruit_contour"
    ]

    detection_image = results.get(
        "detection_image"
    )

    contour_image = create_contour_image(
        working_image,
        fruit_contour
    )

    fruit_only_colour = results[
        "fruit_only_colour"
    ]


    output_images = {
        "working_image.jpg": working_image,
        "otsu_gray_mask.png": gray_mask,
        "combined_otsu_mask.png": combined_mask,
        "otsu_saturation_mask.png": saturation_mask,
        "refined_mask.png": refined_mask,
        "final_fruit_mask.png": fruit_mask,
        "fruit_contour.jpg": contour_image,
        "fruit_only_colour.jpg": fruit_only_colour,
    }

    if detection_image is not None:
        output_images["fruit_detection.jpg"] = detection_image

    for filename, image in output_images.items():
        output_path = os.path.join(output_dir, filename)
        if not cv2.imwrite(output_path, image):
            raise IOError(f"Failed to save result image: {output_path}")

    roi_results = results.get(
        "roi_results",
        []
    )

    for index, roi_result in enumerate(
        roi_results,
        start=1
    ):
        roi_panel = create_roi_visualisation_panel(
            roi_result,
            index
        )

        roi_output_path = os.path.join(
            output_dir,
            f"roi_{index}_panel.jpg"
        )

        if not cv2.imwrite(
            roi_output_path,
            roi_panel
        ):
            raise IOError(
                f"Failed to save ROI panel: "
                f"{roi_output_path}"
            )

    print(
        f"\nResults saved to: {output_dir}"
    )

