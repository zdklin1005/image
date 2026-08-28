from pathlib import Path

import cv2
import numpy as np


def resize_preserving_aspect_ratio(image, target_size=(640, 640)):
    #Resize an image to fit inside target_size without distortion
    target_width, target_height = target_size
    original_height, original_width = image.shape[:2]

    scale = min(
        target_width / original_width,
        target_height / original_height,
    )

    resized_width = max(1, int(round(original_width * scale)))
    resized_height = max(1, int(round(original_height * scale)))

    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized_image = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=interpolation,
    )

    left_padding = (target_width - resized_width) // 2
    top_padding = (target_height - resized_height) // 2
    right_padding = target_width - resized_width - left_padding
    bottom_padding = target_height - resized_height - top_padding

    padding = (
        left_padding,
        top_padding,
        right_padding,
        bottom_padding,
    )

    return resized_image, scale, padding


def add_letterbox_padding(
    image,
    target_size=(640, 640),
    padding_colour=(255, 255, 255),
):
    #Centre an already resized image on a fixed-size canvas.
    target_width, target_height = target_size
    image_height, image_width = image.shape[:2]

    if image_width > target_width or image_height > target_height:
        raise ValueError("Image is larger than the target canvas.")

    standardised_image = np.full(
        (target_height, target_width, 3),
        padding_colour,
        dtype=image.dtype,
    )

    left_padding = (target_width - image_width) // 2
    top_padding = (target_height - image_height) // 2

    standardised_image[
        top_padding:top_padding + image_height,
        left_padding:left_padding + image_width,
    ] = image

    return standardised_image


def create_valid_content_mask(
    output_size,
    resize_padding,
):
    # Create a binary mask separating image content from padding
    output_width, output_height = output_size

    (
        left_padding,
        top_padding,
        right_padding,
        bottom_padding,
    ) = resize_padding

    valid_content_mask = np.zeros(
        (output_height, output_width),
        dtype=np.uint8,
    )

    x1 = left_padding
    y1 = top_padding
    x2 = output_width - right_padding
    y2 = output_height - bottom_padding

    valid_content_mask[y1:y2, x1:x2] = 255

    return valid_content_mask


def calculate_blur_score(image, reference_width=256):
    #Calculate a size-normalised variance-of-Laplacian blur score.
    image_height, image_width = image.shape[:2]
    scale = reference_width / image_width
    reference_height = max(1, int(round(image_height * scale)))

    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    reference_image = cv2.resize(
        image,
        (reference_width, reference_height),
        interpolation=interpolation,
    )

    greyscale_image = cv2.cvtColor(
        reference_image,
        cv2.COLOR_BGR2GRAY,
    )

    return float(
        cv2.Laplacian(
            greyscale_image,
            cv2.CV_64F,
            ksize=3,
        ).var()
    )


def calculate_input_quality_metrics(image):
    #Measure exposure and contrast before enhancement is applied.
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    value_channel = hsv_image[:, :, 2]

    lower_percentile, upper_percentile = np.percentile(
        value_channel,
        (5, 95),
    )

    return {
        "mean_brightness": float(np.mean(value_channel)),
        "contrast_score": float(np.std(value_channel)),
        "dynamic_range": float(upper_percentile - lower_percentile),
        "dark_pixel_ratio": float(np.mean(value_channel <= 40)),
        "bright_pixel_ratio": float(np.mean(value_channel >= 215)),
    }


def preprocess_fruit_image(
    image_path,
    median_kernel=5,
    bilateral_diameter=5,
    bilateral_sigma_colour=25,
    bilateral_sigma_space=25,
    clahe_clip_limit=1.0,
    clahe_grid_size=(8, 8),
    sharpen_strength=0.08,
    blur_threshold=60.0,
    output_size=(640, 640),
):
    #Read, standardise and preprocess one fruit image.
    image_path = Path(image_path)

    if median_kernel <= 0 or median_kernel % 2 == 0:
        raise ValueError("median_kernel must be a positive odd integer.")

    if output_size[0] <= 0 or output_size[1] <= 0:
        raise ValueError("output_size values must be positive integers.")

    # 1. IMAGE READING AND VALIDATION
    source_image = cv2.imread(str(image_path))

    if source_image is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")

    # 2. INPUT-SIZE STANDARDISATION
    # Resize before filtering so every image enters the pipeline at a
    # consistent scale. Padding is added only after processing so it cannot
    # influence denoising, CLAHE or the blur score.
    resized_image, resize_scale, resize_padding = (
        resize_preserving_aspect_ratio(
            source_image,
            output_size,
        )
    )

    # 3. INPUT-QUALITY MEASUREMENT
    # These metrics describe the resized content before filtering or CLAHE.
    # No qualitative exposure labels are assigned until suitable thresholds
    # have been established using the project dataset.
    quality_metrics = calculate_input_quality_metrics(resized_image)

    # 4. MEDIAN DENOISING
    # A fixed 5 x 5 kernel removes the observed salt-and-pepper noise while
    # retaining more detail than the stronger 7 x 7 alternative.
    median_image = cv2.medianBlur(
        resized_image,
        median_kernel,
    )

    # 5. BLUR ASSESSMENT
    blur_score = calculate_blur_score(median_image)
    is_blurry = blur_score < blur_threshold

    # 6. EDGE-PRESERVING BILATERAL DENOISING
    bilateral_image = cv2.bilateralFilter(
        median_image,
        bilateral_diameter,
        bilateral_sigma_colour,
        bilateral_sigma_space,
    )

    # 7. HSV COLOUR-SPACE CONVERSION
    hsv_image = cv2.cvtColor(
        bilateral_image,
        cv2.COLOR_BGR2HSV,
    )
    hue, saturation, value = cv2.split(hsv_image)

    # 8. CLAHE CONTRAST ENHANCEMENT ON THE VALUE CHANNEL
    clahe = cv2.createCLAHE(
        clipLimit=clahe_clip_limit,
        tileGridSize=clahe_grid_size,
    )
    value_enhanced = clahe.apply(value)

    analysis_hsv = cv2.merge(
        (hue, saturation, value_enhanced)
    )
    analysis_image = cv2.cvtColor(
        analysis_hsv,
        cv2.COLOR_HSV2BGR,
    )

    # 9. OPTIONAL LAPLACIAN SHARPENING FOR DISPLAY ONLY
    if sharpen_strength > 0:
        laplacian = cv2.Laplacian(
            value_enhanced,
            cv2.CV_32F,
            ksize=3,
        )

        value_sharpened = np.clip(
            value_enhanced.astype(np.float32)
            - sharpen_strength * laplacian,
            0,
            255,
        ).astype(np.uint8)

        display_hsv = cv2.merge(
            (hue, saturation, value_sharpened)
        )
        display_image = cv2.cvtColor(
            display_hsv,
            cv2.COLOR_HSV2BGR,
        )
    else:
        display_image = analysis_image.copy()

    # 10. FIXED-SIZE OUTPUT CREATION
    # All stages use identical padding and become exactly output_size.
    original_standardised = add_letterbox_padding(
        resized_image,
        output_size,
    )
    median_standardised = add_letterbox_padding(
        median_image,
        output_size,
    )
    bilateral_standardised = add_letterbox_padding(
        bilateral_image,
        output_size,
    )
    analysis_standardised = add_letterbox_padding(
        analysis_image,
        output_size,
    )
    display_standardised = add_letterbox_padding(
        display_image,
        output_size,
    )

    valid_content_mask = create_valid_content_mask(
        output_size,
        resize_padding,
    )

    valid_content_bbox = (
        resize_padding[0],
        resize_padding[1],
        output_size[0] - resize_padding[2],
        output_size[1] - resize_padding[3],
    )

    return {
        "original_image": original_standardised,
        "median_image": median_standardised,
        "bilateral_image": bilateral_standardised,
        "classification_image": bilateral_standardised,
        "analysis_image": analysis_standardised,
        "display_image": display_standardised,
        "source_image_full_resolution": source_image,
        "resize_scale": resize_scale,
        "resize_padding": resize_padding,
        "output_size": output_size,
        "valid_content_mask": valid_content_mask,
        "valid_content_bbox": valid_content_bbox,
        "blur_score": blur_score,
        "is_blurry": is_blurry,
        **quality_metrics,
    }
