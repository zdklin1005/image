import cv2
import numpy as np


def preprocess_fruit_image(image_path, sharpen_strength=0.15):
    # 1. Read and validate the input image
    raw_img = cv2.imread(str(image_path))

    if raw_img is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")

    # 2. Remove impulse noise
    median_filtered = cv2.medianBlur(raw_img, 3)

    # 3. Perform edge-preserving smoothing
    bilateral_filtered = cv2.bilateralFilter(median_filtered, d=5, sigmaColor=25, sigmaSpace=25)

    # 4. Convert to HSV and enhance only the Value channel
    hsv_img = cv2.cvtColor(bilateral_filtered, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_img)

    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    v_enhanced = clahe.apply(v)

    # 5. Laplacian sharpening using the signed Laplacian
    laplacian = cv2.Laplacian(v_enhanced, cv2.CV_32F, ksize=3)

    v_sharpened = np.clip(v_enhanced.astype(np.float32)- sharpen_strength * laplacian, 0, 255).astype(np.uint8)

    # 6. Reconstruct the output image
    hsv_output = cv2.merge((h, s, v_sharpened))
    final_output = cv2.cvtColor(hsv_output, cv2.COLOR_HSV2BGR)

    return final_output

#replace with ur own input path
input_path = (
    r"C:\Users\Kai\Documents\Image Processing"
    r"\assgm\dataset\Test\freshapples\a_f326.png"
)

clean_image = preprocess_fruit_image(input_path)

if not cv2.imwrite("output_fresh_apple.jpg", clean_image):
    raise IOError("Failed to save the processed image.")