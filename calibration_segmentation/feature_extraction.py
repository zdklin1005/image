import cv2
import numpy as np


def extract_colour_features(
    image,
    fruit_mask
):
    """
    Extract colour statistics from the detected fruit region.

    Only pixels inside fruit_mask are analysed.

    Parameters:
        image:
            BGR colour image.

        fruit_mask:
            Binary mask where fruit pixels are white (255).

    Returns:
        Dictionary containing BGR and HSV mean values.
    """

    if image is None:
        raise ValueError(
            "Input image cannot be None."
        )

    if fruit_mask is None:
        raise ValueError(
            "Fruit mask cannot be None."
        )

    if image.shape[:2] != fruit_mask.shape[:2]:
        raise ValueError(
            "Image and fruit mask dimensions must match."
        )

    if cv2.countNonZero(fruit_mask) == 0:
        raise ValueError(
            "Fruit mask contains no foreground pixels."
        )

    # --------------------------------------------------------
    # BGR colour features
    # --------------------------------------------------------

    mean_b, mean_g, mean_r, _ = cv2.mean(
        image,
        mask=fruit_mask
    )

    # --------------------------------------------------------
    # HSV colour features
    # --------------------------------------------------------

    hsv_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    hue_channel = hsv_image[:, :, 0]

    hue_histogram = cv2.calcHist(
        [hue_channel],
        [0],
        fruit_mask,
        [180],
        [0, 180]
    )

    dominant_hue = int(
        np.argmax(hue_histogram)
    )

    _, mean_s, mean_v, _ = cv2.mean(
        hsv_image,
        mask=fruit_mask
    )

    return {
        "mean_blue": mean_b,
        "mean_green": mean_g,
        "mean_red": mean_r,

        "dominant_hue": dominant_hue,
        "mean_saturation": mean_s,
        "mean_value": mean_v
    }