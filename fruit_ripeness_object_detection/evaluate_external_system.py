"""Evaluate the integrated fruit pipeline on external multi-fruit images."""

import csv
import json
from pathlib import Path

import cv2

from calibration_segmentation.roi_processing import process_fruit_roi
from preprocessing import preprocess_fruit_image
from fruit_ripeness_object_detection.blemish import detect_fruit_blemish
from fruit_ripeness_object_detection.fruit_detection import (
    detect_with_model_a,
    detect_with_model_c,
    detect_with_model_d,
    draw_final_detections,
    fuse_detections,
)
from fruit_ripeness_object_detection.ripeness_classification import (
    classify_with_model_b,
    classify_with_model_e,
    fuse_ripeness,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIRECTORY = PROJECT_ROOT / "external_system_evaluation"
OUTPUT_DIRECTORY = DEFAULT_INPUT_DIRECTORY / "outputs"
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png"}


def evaluate_image(image_path):
    preprocessing = preprocess_fruit_image(image_path)
    classification_image = preprocessing["classification_image"]
    analysis_image = preprocessing["analysis_image"]

    detections_a = detect_with_model_a(classification_image, 0.30)
    detections_c = detect_with_model_c(classification_image, 0.30)
    detections_d = detect_with_model_d(classification_image, 0.30)
    final_detections = fuse_detections(
        detections_a,
        detections_c,
        detections_d,
        iou_threshold=0.30,
    )

    image_rows = []

    for detection_index, detection in enumerate(final_detections, start=1):
        row = {
            "image": image_path.name,
            "detection_index": detection_index,
            "fruit_type": detection["fruit_type"],
            "detection_confidence": detection["confidence"],
            "bounding_box": str(detection["bounding_box"]),
            "agreement": detection["agreement"],
            "class_disagreement": detection.get("class_disagreement", False),
            "class_votes": json.dumps(detection.get("class_votes", {})),
            "model_c_ripeness": detection.get("model_c_ripeness"),
            "final_ripeness": None,
            "ripeness_confidence": None,
            "fruit_area_pixels": None,
            "blemish_percentage": None,
            "roi_error": None,
        }

        if not detection.get("is_supported", True):
            row["roi_error"] = (
                "Skipped: unsupported detected class "
                f"{detection.get('detected_fruit_type', 'Unknown')}"
            )
            image_rows.append(row)
            continue

        try:
            roi = process_fruit_roi(
                analysis_image,
                detection["bounding_box"],
                use_watershed=False,
            )
            x1, y1, x2, y2 = roi["bounding_box"]
            fruit_roi = analysis_image[y1:y2, x1:x2].copy()
            result_b = classify_with_model_b(
                fruit_roi,
                detection["fruit_type"],
            )
            result_e = classify_with_model_e(fruit_roi)
            final_ripeness = fuse_ripeness(
                result_b,
                detection.get("model_c_ripeness"),
                detection.get("confidence_c"),
                result_e,
            )
            blemish = detect_fruit_blemish(
                roi["roi_image"],
                roi["fruit_mask"],
                detection["fruit_type"],
            )

            row.update({
                "final_ripeness": final_ripeness["ripeness"],
                "ripeness_confidence": final_ripeness["confidence"],
                "fruit_area_pixels": roi["fruit_area_pixels"],
                "blemish_percentage": blemish["blemish_percentage"],
            })
        except Exception as error:
            row["roi_error"] = f"{type(error).__name__}: {error}"

        image_rows.append(row)

    annotated = draw_final_detections(
        classification_image,
        final_detections,
    )
    return preprocessing, final_detections, image_rows, annotated


def main():
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(
        path
        for path in DEFAULT_INPUT_DIRECTORY.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    all_rows = []
    image_summaries = []

    for image_path in image_paths:
        preprocessing, detections, rows, annotated = evaluate_image(image_path)
        all_rows.extend(rows)
        output_path = OUTPUT_DIRECTORY / f"{image_path.stem}_detections.jpg"
        cv2.imwrite(str(output_path), annotated)
        image_summaries.append({
            "image": image_path.name,
            "final_groups_with_model_a": sum(
                "A" in item["models"] for item in detections
            ),
            "final_groups_with_model_c": sum(
                "C" in item["models"] for item in detections
            ),
            "final_groups_with_model_d": sum(
                "D" in item["models"] for item in detections
            ),
            "final_detection_count": len(detections),
            "predicted_classes": [item["fruit_type"] for item in detections],
            "blur_score": preprocessing["blur_score"],
            "is_blurry": bool(preprocessing["is_blurry"]),
        })
        print(
            image_path.name,
            [(item["fruit_type"], round(item["confidence"], 3)) for item in detections],
            flush=True,
        )

    csv_path = OUTPUT_DIRECTORY / "external_detection_results.csv"
    fieldnames = [
        "image",
        "detection_index",
        "fruit_type",
        "detection_confidence",
        "bounding_box",
        "agreement",
        "class_disagreement",
        "class_votes",
        "model_c_ripeness",
        "final_ripeness",
        "ripeness_confidence",
        "fruit_area_pixels",
        "blemish_percentage",
        "roi_error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    summary_path = OUTPUT_DIRECTORY / "external_detection_summary.json"
    summary_path.write_text(
        json.dumps(image_summaries, indent=2),
        encoding="utf-8",
    )

    print(f"Results saved to {OUTPUT_DIRECTORY}")


if __name__ == "__main__":
    main()
