import cv2
import numpy as np


# ============================================================
# TECHNIQUE 1:
# CALIBRATION REFERENCE LOCALISATION
# ============================================================

def select_reference_points(image):
    """
    Allow the user to manually select the four corner
    points of a known rectangular calibration reference.

    Click order:
        1. Top-left
        2. Top-right
        3. Bottom-right
        4. Bottom-left

    Parameters:
        image:
            Image used for displaying and selecting
            calibration points.

    Returns:
        points:
            Four selected corner coordinates as
            a NumPy float32 array.
    """

    if image is None:
        raise ValueError(
            "Input image for reference selection cannot be None."
        )

    points = []
    display_image = image.copy()

    window_name = "Select 4 Calibration Points"

    def mouse_callback(event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:

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
                    f"Point {len(points)} selected: "
                    f"({x}, {y})"
                )

                cv2.imshow(
                    window_name,
                    display_image
                )

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

    while len(points) < 4:

        key = cv2.waitKey(20) & 0xFF

        if key == 27:

            cv2.destroyAllWindows()

            raise ValueError(
                "Calibration point selection cancelled."
            )

    cv2.destroyAllWindows()

    return np.float32(points)


# ============================================================
# TECHNIQUE 2:
# PERSPECTIVE TRANSFORMATION / RECTIFICATION
# ============================================================

def rectify_perspective(
    image,
    points,
    reference_width_cm,
    reference_height_cm,
    target_pixels_per_cm=20
):

    if image is None:
        raise ValueError(
            "Input image cannot be None."
        )

    points = np.asarray(
        points,
        dtype=np.float32
    )

    if points.shape != (4, 2):
        raise ValueError(
            "Exactly four 2D calibration points are required."
        )

    if (
        reference_width_cm <= 0
        or reference_height_cm <= 0
    ):
        raise ValueError(
            "Reference dimensions must be greater than zero."
        )

    if target_pixels_per_cm <= 0:
        raise ValueError(
            "target_pixels_per_cm must be greater than zero."
        )

    # --------------------------------------------------------
    # Calculate rectified output dimensions
    # --------------------------------------------------------

    output_width = int(
        round(
            reference_width_cm
            * target_pixels_per_cm
        )
    )

    output_height = int(
        round(
            reference_height_cm
            * target_pixels_per_cm
        )
    )

    if output_width < 2 or output_height < 2:
        raise ValueError(
            "Calculated rectified image size is too small."
        )

    # --------------------------------------------------------
    # Destination coordinates
    # --------------------------------------------------------

    destination_points = np.float32([
        [0, 0],
        [output_width - 1, 0],
        [
            output_width - 1,
            output_height - 1
        ],
        [0, output_height - 1]
    ])

    # --------------------------------------------------------
    # Perspective transformation
    # --------------------------------------------------------

    transformation_matrix = (
        cv2.getPerspectiveTransform(
            points,
            destination_points
        )
    )

    rectified_image = cv2.warpPerspective(
        image,
        transformation_matrix,
        (
            output_width,
            output_height
        )
    )

    return (
        rectified_image,
        transformation_matrix
    )


# ============================================================
# TECHNIQUE 3:
# REFERENCE OBJECT SPATIAL CALIBRATION
# ============================================================

def calculate_spatial_calibration(
    rectified_image,
    reference_width_cm,
    reference_height_cm
):

    if rectified_image is None:
        raise ValueError(
            "Rectified image cannot be None."
        )

    if (
        reference_width_cm <= 0
        or reference_height_cm <= 0
    ):
        raise ValueError(
            "Reference dimensions must be greater than zero."
        )

    height_px, width_px = (
        rectified_image.shape[:2]
    )

    pixels_per_cm_x = (
        width_px
        / reference_width_cm
    )

    pixels_per_cm_y = (
        height_px
        / reference_height_cm
    )

    average_scale = (
        pixels_per_cm_x
        + pixels_per_cm_y
    ) / 2

    scale_difference = abs(
        pixels_per_cm_x
        - pixels_per_cm_y
    )

    difference_percentage = (
        scale_difference
        / average_scale
    ) * 100

    return (
        pixels_per_cm_x,
        pixels_per_cm_y,
        difference_percentage
    )


# ============================================================
# COMPLETE CALIBRATION PIPELINE
# ============================================================

def calibrate_image(
    analysis_image,
    reference_width_cm,
    reference_height_cm,
    target_pixels_per_cm=20,
    selection_image=None
):

    if selection_image is None:
        selection_image = analysis_image

    # --------------------------------------------------------
    # Technique 1:
    # Calibration Reference Localisation
    # --------------------------------------------------------

    reference_points = select_reference_points(
        selection_image
    )

    # --------------------------------------------------------
    # Technique 2:
    # Perspective Rectification
    # --------------------------------------------------------

    (
        rectified_image,
        transformation_matrix
    ) = rectify_perspective(
        analysis_image,
        reference_points,
        reference_width_cm,
        reference_height_cm,
        target_pixels_per_cm
    )

    # --------------------------------------------------------
    # Technique 3:
    # Spatial Calibration
    # --------------------------------------------------------

    (
        pixels_per_cm_x,
        pixels_per_cm_y,
        difference_percentage
    ) = calculate_spatial_calibration(
        rectified_image,
        reference_width_cm,
        reference_height_cm
    )

    return {
        "rectified_image":
            rectified_image,

        "transformation_matrix":
            transformation_matrix,

        "reference_points":
            reference_points,

        "pixels_per_cm_x":
            pixels_per_cm_x,

        "pixels_per_cm_y":
            pixels_per_cm_y,

        "scale_difference_percentage":
            difference_percentage
    }