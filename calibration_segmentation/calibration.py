import cv2
import numpy as np
import os

from calibration_segmentation.segmentation import (
    segment_fruit_otsu,
    refine_fruit_mask,
)

from calibration_segmentation.measurement import (
    extract_main_fruit,
    calculate_projected_area_cm2
)

# ============================================================
# IMAGE LOADING
# ============================================================

def load_image(image_path):
    """
    Load an image for the calibration and segmentation module.
    """

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(
            f"Unable to load image: {image_path}"
        )

    return image


# ============================================================
# CALIBRATION REFERENCE LOCALISATION
# ============================================================

def select_reference_points(image):
    """
    Allow the user to manually select four corner points
    of the calibration reference.

    Click order:
    1. Top-left
    2. Top-right
    3. Bottom-right
    4. Bottom-left
    """

    points = []
    display_image = image.copy()

    window_name = "Select 4 Calibration Points"

    def mouse_callback(event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:

            # Only allow four points
            if len(points) < 4:

                points.append((x, y))

                # Draw selected point
                cv2.circle(
                    display_image,
                    (x, y),
                    6,
                    (0, 0, 255),
                    -1
                )

                # Display point number
                cv2.putText(
                    display_image,
                    str(len(points)),
                    (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

                print(
                    f"Point {len(points)} selected: ({x}, {y})"
                )

                cv2.imshow(
                    window_name,
                    display_image
                )

    # Create resizable window
    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL
    )

    cv2.imshow(
        window_name,
        display_image
    )

    cv2.setMouseCallback(
        window_name,
        mouse_callback
    )

    print("\nSelect the four calibration points:")
    print("1. Top-left")
    print("2. Top-right")
    print("3. Bottom-right")
    print("4. Bottom-left")
    print("Press ESC to cancel.")

    # Wait until all four points are selected
    while len(points) < 4:

        key = cv2.waitKey(20) & 0xFF

        # ESC key cancels point selection
        if key == 27:

            cv2.destroyAllWindows()

            raise ValueError(
                "Calibration point selection cancelled."
            )

    cv2.destroyAllWindows()

    if len(points) != 4:
        raise ValueError(
            "Exactly 4 calibration points must be selected."
        )

    return np.float32(points)


def rectify_perspective(image, points):
    """
    Correct perspective distortion using four selected points.

    Point order:
    1. Top-left
    2. Top-right
    3. Bottom-right
    4. Bottom-left
    """

    # Extract the four points
    top_left = points[0]
    top_right = points[1]
    bottom_right = points[2]
    bottom_left = points[3]

    # --------------------------------------------------------
    # Calculate output width
    # --------------------------------------------------------

    top_width = np.linalg.norm(
        top_right - top_left
    )

    bottom_width = np.linalg.norm(
        bottom_right - bottom_left
    )

    output_width = int(
        max(top_width, bottom_width)
    )

    # --------------------------------------------------------
    # Calculate output height
    # --------------------------------------------------------

    left_height = np.linalg.norm(
        bottom_left - top_left
    )

    right_height = np.linalg.norm(
        bottom_right - top_right
    )

    output_height = int(
        max(left_height, right_height)
    )

    if output_width < 2 or output_height < 2:
        raise ValueError(
            "Selected calibration region is too small."
        )
    # --------------------------------------------------------
    # Destination points
    # --------------------------------------------------------

    destination_points = np.float32([
        [0, 0],
        [output_width - 1, 0],
        [output_width - 1, output_height - 1],
        [0, output_height - 1]
    ])

    # --------------------------------------------------------
    # Calculate perspective transformation matrix
    # --------------------------------------------------------

    transformation_matrix = cv2.getPerspectiveTransform(
        points,
        destination_points
    )

    # --------------------------------------------------------
    # Perform perspective transformation
    # --------------------------------------------------------

    rectified_image = cv2.warpPerspective(
        image,
        transformation_matrix,
        (output_width, output_height)
    )

    return rectified_image, transformation_matrix

def calculate_spatial_calibration(
    rectified_image,
    reference_width_cm,
    reference_height_cm
):
    """
    Calculate pixel-to-centimetre spatial calibration
    using a known rectangular reference.
    """

    if reference_width_cm <= 0 or reference_height_cm <= 0:
        raise ValueError(
            "Reference dimensions must be greater than zero."
        )

    height_px, width_px = rectified_image.shape[:2]

    pixels_per_cm_x = (
        width_px / reference_width_cm
    )

    pixels_per_cm_y = (
        height_px / reference_height_cm
    )

    average_scale = (
        pixels_per_cm_x + pixels_per_cm_y
    ) / 2

    scale_difference = abs(
        pixels_per_cm_x - pixels_per_cm_y
    )

    difference_percentage = (
        scale_difference / average_scale
    ) * 100

    print("\nSpatial Calibration")
    print("------------------------------")

    print(
        f"Reference width  : "
        f"{reference_width_cm:.2f} cm"
    )

    print(
        f"Reference height : "
        f"{reference_height_cm:.2f} cm"
    )

    print(
        f"Rectified width  : "
        f"{width_px} pixels"
    )

    print(
        f"Rectified height : "
        f"{height_px} pixels"
    )

    print(
        f"Horizontal scale : "
        f"{pixels_per_cm_x:.2f} pixels/cm"
    )

    print(
        f"Vertical scale   : "
        f"{pixels_per_cm_y:.2f} pixels/cm"
    )

    print(
        f"Scale difference : "
        f"{difference_percentage:.2f}%"
    )

    if difference_percentage <= 5:
        print("Calibration status: GOOD")

    else:
        print(
            "Calibration status: WARNING - "
            "horizontal and vertical scales differ significantly."
        )

    return pixels_per_cm_x, pixels_per_cm_y

# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    image_path = "a_f326.png"

    # Step 1: Load image
    # TODO:
    # Replace direct image loading with Member 1 preprocessing output
    # during final module integration.
    image = load_image(image_path)

    # Step 2: Calibration Reference Localisation
    reference_points = select_reference_points(image)

    print("\nSelected Calibration Points:")
    print(reference_points)

    # Step 3: Perspective Rectification
    rectified_image, transformation_matrix = rectify_perspective(
        image,
        reference_points
    )

    print("\nPerspective Transformation Matrix:")
    print(transformation_matrix)

    # ========================================================
    # Technique 3: Reference Object Spatial Calibration
    # ========================================================

    # TEMPORARY VALUES FOR DEVELOPMENT ONLY
    reference_width_cm = 30.0
    reference_height_cm = 20.0

    pixels_per_cm_x, pixels_per_cm_y = (
        calculate_spatial_calibration(
            rectified_image,
            reference_width_cm,
            reference_height_cm
        )
    )

    # ========================================================
    # Technique 4: Otsu's Binarisation
    # ========================================================

    (
        gray_image,
        gray_mask,
        gray_threshold,
        saturation_image,
        saturation_mask,
        saturation_threshold
    ) = segment_fruit_otsu(rectified_image)


    print("\nOtsu Segmentation")
    print("------------------------------")

    print(
        f"Grayscale Otsu threshold  : "
        f"{gray_threshold:.2f}"
    )

    print(
        f"Saturation Otsu threshold : "
        f"{saturation_threshold:.2f}"
    )

    

    # ========================================================
    # Technique 5: Morphological Refinement
    # ========================================================

    opening_kernel_size = 3
    closing_kernel_size = 5

    opened_mask, refined_mask = refine_fruit_mask(
        saturation_mask,
        opening_kernel_size,
        closing_kernel_size
    )

    print("\nMorphological Refinement")
    print("------------------------------")

    print(
        f"Opening kernel : "
        f"{opening_kernel_size} x {opening_kernel_size}"
    )

    print(
        f"Closing kernel : "
        f"{closing_kernel_size} x {closing_kernel_size}"
    )

    print("Opening        : Applied")
    print("Closing        : Applied")

    # ========================================================
    # Technique 6: Watershed
    # ========================================================

    # OPTIONAL:
    # Watershed is used only when multiple foreground fruits
    # are touching each other.
    #
    # The current test image contains only one fruit,
    # therefore Watershed is skipped.


    # ========================================================
    # Technique 7: Calibrated Projected Fruit Area Measurement
    # ========================================================

    (
        fruit_mask,
        fruit_contour,
        fruit_area_pixels
    ) = extract_main_fruit(
        refined_mask
    )

    fruit_area_cm2 = calculate_projected_area_cm2(
        fruit_area_pixels,
        pixels_per_cm_x,
        pixels_per_cm_y
    )

    print(
        f"Fruit area in pixels : "
        f"{fruit_area_pixels} pixels"
    )

    print(
        f"Projected fruit area : "
        f"{fruit_area_cm2:.2f} cm^2"
    )

    print(
        "\nNOTE: Physical area is currently for "
        "development/testing only because temporary "
        "reference dimensions are being used."
    )


    # --------------------------------------------------------
    # Create fruit contour visualisation
    # --------------------------------------------------------

    contour_image = rectified_image.copy()

    cv2.drawContours(
        contour_image,
        [fruit_contour],
        -1,
        (0, 0, 255),
        2
    )

    # ========================================================
    # Display Results
    # ========================================================

    cv2.namedWindow(
        "Rectified Image",
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
        "Fruit Area Contour",
        cv2.WINDOW_NORMAL
    )


    cv2.imshow(
        "Rectified Image",
        rectified_image
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
        "Fruit Area Contour",
        contour_image
    )


    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # ========================================================
    # Save Results
    # ========================================================
    
    output_dir = "calibration_segmentation/outputs"

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    cv2.imwrite(
        f"{output_dir}/rectified_image.jpg",
        rectified_image
    )

    cv2.imwrite(
        f"{output_dir}/grayscale_image.png",
        gray_image
    )

    cv2.imwrite(
        f"{output_dir}/otsu_gray_mask.png",
        gray_mask
    )

    cv2.imwrite(
        f"{output_dir}/saturation_channel.png",
        saturation_image
    )

    cv2.imwrite(
        f"{output_dir}/otsu_saturation_mask.png",
        saturation_mask
    )

    cv2.imwrite(
        f"{output_dir}/morph_opening.png",
        opened_mask
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
        f"{output_dir}/fruit_area_contour.jpg",
        contour_image
    )


    print("\nImages successfully saved.")