"""Run one selected detection through each downstream stage with traceback."""

import argparse
import traceback

from calibration_segmentation.roi_processing import process_fruit_roi
from fruit_ripeness_object_detection.blemish import detect_fruit_blemish
from fruit_ripeness_object_detection.fruit_detection import (
    assess_detection_quality,
    detect_with_model_a,
    detect_with_model_c,
    detect_with_model_d,
    fuse_detections,
)
from fruit_ripeness_object_detection.ripeness_classification import (
    classify_with_model_b,
    classify_with_model_e,
    fuse_ripeness,
)
from preprocessing import preprocess_fruit_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("detection_index", type=int, help="One-based index")
    args = parser.parse_args()

    preprocessing = preprocess_fruit_image(args.image)
    image = preprocessing["classification_image"]
    detections = fuse_detections(
        detect_with_model_a(image, 0.30),
        detect_with_model_c(image, 0.30),
        detect_with_model_d(image, 0.30),
        iou_threshold=0.30,
    )
    detections = assess_detection_quality(
        detections,
        image.shape,
        valid_content_bbox=preprocessing["valid_content_bbox"],
        retain_rejected=True,
    )
    detection = detections[args.detection_index - 1]
    print("Detection:", detection)

    stage = "ROI segmentation"
    try:
        roi = process_fruit_roi(
            preprocessing["analysis_image"],
            detection["bounding_box"],
            use_watershed=False,
        )
        x1, y1, x2, y2 = roi["bounding_box"]
        fruit_roi = preprocessing["analysis_image"][y1:y2, x1:x2].copy()

        stage = "Model B classification"
        result_b = classify_with_model_b(fruit_roi, detection["fruit_type"])
        stage = "Model E classification"
        result_e = classify_with_model_e(fruit_roi)
        stage = "ripeness fusion"
        ripeness = fuse_ripeness(
            result_b,
            detection.get("model_c_ripeness"),
            detection.get("confidence_c"),
            result_e,
        )
        print("Ripeness:", ripeness)

        stage = "blemish analysis"
        blemish = detect_fruit_blemish(
            roi["roi_image"], roi["fruit_mask"], detection["fruit_type"]
        )
        print("Blemish percentage:", blemish["blemish_percentage"])
        print("All downstream stages passed.")
    except Exception:
        print(f"FAILED STAGE: {stage}")
        traceback.print_exc()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
