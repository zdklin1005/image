import cv2
import numpy as np

def preprocess_fruit_image(image_path):
    # 1. Read the raw BGR image
    raw_img = cv2.imread(image_path)
    
    # 2. Colour Space Conversion (BGR to HSV)
    hsv_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_img) # Isolate the Value (Luminance) channel
    
    # 3. Contrast Enhancement (CLAHE on V-channel)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
    v_clahe = clahe.apply(v)
    
    # Merge channels back and convert to BGR for spatial filtering
    hsv_merged = cv2.merge([h, s, v_clahe])
    bgr_clahe = cv2.cvtColor(hsv_merged, cv2.COLOR_HSV2BGR)
    
    # 4. Image Denoising 1: Median Filter (Removes impulse noise)
    # Kernel size 3x3 is standard for light sensor grain
    median_filtered = cv2.medianBlur(bgr_clahe, 3)
    
    # 5. Image Denoising 2: Bilateral Filter (Edge-preserving smoothing)
    # 5 = neighborhood diameter, 25 = sigma color, 25 = sigma space
    bilateral_filtered = cv2.bilateralFilter(median_filtered, 5, 25, 25)
    
    # 6. High-Pass Filter: Laplacian Sharpening
    # Calculate Laplacian edges, then subtract them to sharpen the original
    laplacian_edges = cv2.Laplacian(bilateral_filtered, cv2.CV_64F)
    laplacian_edges = cv2.convertScaleAbs(laplacian_edges)
    
    # Add the edges back to the smoothed image to crisp the defect boundaries
    final_output = cv2.addWeighted(bilateral_filtered, 1.0, laplacian_edges, -0.2, 0)
    
    return final_output

# Example Execution:
clean_image = preprocess_fruit_image(r"C:\Users\Kai\Documents\Image Processing\assgm\dataset\Test\freshapples\a_f326.png")
cv2.imwrite("output_banana.jpg", clean_image)