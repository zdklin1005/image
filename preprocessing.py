from pathlib import Path

import cv2
import numpy as np


def calculate_blur_score(image, reference_width=256):
    # BLUR ANALYSIS:
    # Resize the image to make blur scores more comparable between images.
    height, width = image.shape[:2]
    scale = reference_width / width
    reference_height = max(1, int(height * scale))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR

    resized = cv2.resize(image, (reference_width, reference_height), interpolation=interpolation)
    grey = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # A lower value normally indicates a blurrier image.
    # A higher value normally indicates a sharper image.
    blur_score = float(cv2.Laplacian(grey, cv2.CV_64F, ksize=3).var())

    return blur_score


def preprocess_fruit_image(image_path, median_kernel=3, bilateral_diameter=5, bilateral_sigma_colour=25, bilateral_sigma_space=25, clahe_clip_limit=1.0, clahe_grid_size=(8, 8), sharpen_strength=0.08, blur_threshold=60.0):

    image_path = Path(image_path)

    # IMAGE READING:
    # Read the original image from its file path.
    raw_image = cv2.imread(str(image_path))

    if raw_image is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")

    # DENOISING PART 1:
    # Median filtering removes salt-and-pepper noise.
    median_image = cv2.medianBlur(raw_image, median_kernel)

    # BLUR ANALYSIS:
    # Analyse the image after removing impulse noise.
    blur_score = calculate_blur_score(median_image)

    # BLUR SUITABILITY DECISION:
    # Determine whether the image is suitable for further processing.
    is_blurry = blur_score < blur_threshold

    # DENOISING PART 2:
    # Bilateral filtering performs additional smoothing while preserving edges.
    bilateral_image = cv2.bilateralFilter(median_image, bilateral_diameter, bilateral_sigma_colour, bilateral_sigma_space)

    # COLOUR-SPACE CONVERSION:
    # Convert BGR to HSV so brightness can be processed separately.
    hsv_image = cv2.cvtColor(bilateral_image, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv_image)

    # CONTRAST ENHANCEMENT:
    # Apply CLAHE only to the Value/brightness channel.
    clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_grid_size)
    value_enhanced = clahe.apply(value)

    # NON-SHARPENED ANALYSIS OUTPUT:
    analysis_hsv = cv2.merge((hue, saturation, value_enhanced))
    analysis_image = cv2.cvtColor(analysis_hsv, cv2.COLOR_HSV2BGR)

    # DISPLAY-IMAGE SHARPENING:
    if sharpen_strength > 0:
        laplacian = cv2.Laplacian(value_enhanced, cv2.CV_32F, ksize=3)
        value_sharpened = np.clip(value_enhanced.astype(np.float32) - sharpen_strength * laplacian, 0, 255).astype(np.uint8)
        display_hsv = cv2.merge((hue, saturation, value_sharpened))
        display_image = cv2.cvtColor(display_hsv, cv2.COLOR_HSV2BGR)
    else:
        display_image = analysis_image.copy()

    # RETURN RESULTS:
    # Return the processed images, blur score and suitability decision.
    return {
        "original_image": raw_image,
        "median_image": median_image,
        "bilateral_image": bilateral_image,
        "analysis_image": analysis_image,
        "display_image": display_image,
        "blur_score": blur_score,
        "is_blurry": is_blurry
    }


def save_image(output_path, image):
    # FILE CREATION:
    output_path = Path(output_path)

    if not cv2.imwrite(str(output_path), image):
        raise IOError(f"Failed to save image: {output_path}")


def save_preprocessing_results(results, output_directory):
    # OUTPUT-FOLDER CREATION:
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    # INDIVIDUAL OUTPUT-FILE CONFIGURATION:
    stages = [
        ("01_original.png", results["original_image"]),
        ("02_median.png", results["median_image"]),
        ("03_bilateral.png", results["bilateral_image"]),
        ("04_analysis_clahe.png", results["analysis_image"]),
        ("05_display_sharpened.png", results["display_image"])
    ]

    # CREATE EACH PROCESSING-STAGE FILE:
    for filename, image in stages:
        save_image(output_directory / filename, image)

#need to alter this section abit to ensure able to run and integrate with ur part, if standalone then no need
if __name__ == "__main__":
    # INPUT-FILE LOCATION:
    # Change this path to select another fruit image.
    input_path = Path("C:/Users/Kai/Documents/Image Processing/assgm/dataset/Test/freshapples/saltandpepper_Screen Shot 2018-06-08 at 5.04.48 PM.png")

    # OUTPUT-FOLDER LOCATION:
    # Results are created inside preprocessing_results beside this Python file.
    script_directory = Path(__file__).resolve().parent
    output_directory = script_directory / "preprocessing_results"

    #this must keep for the whole thing to run
    # RUN THE PREPROCESSING PIPELINE:
    results = preprocess_fruit_image(
        image_path=input_path,
        median_kernel=3,
        bilateral_diameter=5,
        bilateral_sigma_colour=25,
        bilateral_sigma_space=25,
        clahe_clip_limit=1.0,
        clahe_grid_size=(8, 8),
        sharpen_strength=0.08,
        blur_threshold=60.0
    )

    # CREATE THE INDIVIDUAL OUTPUT FILES:
    save_preprocessing_results(results, output_directory)

    # DISPLAY VALUES IN THE TERMINAL:
    print("Preprocessing completed successfully.")
    print(f"Blur score: {results['blur_score']:.2f}")

    # DISPLAY THE BLUR SUITABILITY DECISION IN THE TERMINAL:
    if results["is_blurry"]:
        print("Warning: The image may be too blurry for reliable fruit analysis.")
    else:
        print("Image sharpness is acceptable.")

    # DISPLAY THE OUTPUT-FOLDER LOCATION IN THE TERMINAL:
    print(f"Results saved in: {output_directory}")