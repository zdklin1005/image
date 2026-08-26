import cv2
import os


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

    cv2.waitKey(0)
    cv2.destroyAllWindows()


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

    contour_image = create_contour_image(
        working_image,
        fruit_contour
    )


    cv2.imwrite(
        f"{output_dir}/working_image.jpg",
        working_image
    )

    cv2.imwrite(
        f"{output_dir}/otsu_gray_mask.png",
        gray_mask
    )

    cv2.imwrite(
        f"{output_dir}/combined_otsu_mask.png",
        combined_mask
    )

    cv2.imwrite(
        f"{output_dir}/otsu_saturation_mask.png",
        saturation_mask
    )

    cv2.imwrite(
        f"{output_dir}/refined_mask.png",
        refined_mask
    )

    cv2.imwrite(
        f"{output_dir}/final_fruit_mask.png",
        fruit_mask
    )

    cv2.imwrite(
        f"{output_dir}/fruit_contour.jpg",
        contour_image
    )

    print(
        f"\nResults saved to: {output_dir}"
    )