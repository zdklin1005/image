"""Compare raw and fruit-matched Model C ripeness fusion on fixed samples."""

import csv
import json
from pathlib import Path

from calibration_segmentation.roi_processing import process_fruit_roi
from preprocessing import preprocess_fruit_image
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


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_CSV = PROJECT_ROOT / "preprocessing_evaluation" / "archive_tuning_summary.csv"
OUTPUT_CSV = PROJECT_ROOT / "preprocessing_evaluation" / "model_gate_regression.csv"
OUTPUT_JSON = PROJECT_ROOT / "preprocessing_evaluation" / "model_gate_regression.json"


def load_samples():
    samples = {}

    with SOURCE_CSV.open(encoding="utf-8-sig", newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            samples[row["image"]] = {
                "image": Path(row["image"]),
                "expected_fruit": row["expected_fruit"],
                "expected_ripeness": row["expected_ripeness"],
            }

    return sorted(samples.values(), key=lambda item: str(item["image"]))


def evaluate_sample(sample):
    preprocessing = preprocess_fruit_image(sample["image"])
    classification_image = preprocessing["classification_image"]
    analysis_image = preprocessing["analysis_image"]
    final_detections = fuse_detections(
        detect_with_model_a(classification_image, 0.30),
        detect_with_model_c(classification_image, 0.30),
        detect_with_model_d(classification_image, 0.30),
    )
    final_detections = assess_detection_quality(
        final_detections,
        classification_image.shape,
        valid_content_bbox=preprocessing["valid_content_bbox"],
    )

    if not final_detections:
        return {"status": "no_detection"}

    detection = max(final_detections, key=lambda item: item["confidence"])

    if detection["fruit_type"].casefold() != sample["expected_fruit"].casefold():
        return {
            "status": "wrong_fruit",
            "detected_fruit": detection["fruit_type"],
        }

    try:
        roi = process_fruit_roi(
            analysis_image,
            detection["bounding_box"],
            use_watershed=False,
        )
        x1, y1, x2, y2 = roi["bounding_box"]
        fruit_roi = analysis_image[y1:y2, x1:x2].copy()
        result_b = classify_with_model_b(fruit_roi, detection["fruit_type"])
        result_e = classify_with_model_e(fruit_roi)
        raw_fusion = fuse_ripeness(
            result_b,
            detection.get("model_c_raw_ripeness"),
            detection.get("confidence_c"),
            result_e,
        )
        gated_fusion = fuse_ripeness(
            result_b,
            detection.get("model_c_ripeness"),
            detection.get("confidence_c"),
            result_e,
        )
    except Exception as error:
        return {"status": f"roi_error: {type(error).__name__}: {error}"}

    expected = sample["expected_ripeness"]
    return {
        "status": "evaluated",
        "detected_fruit": detection["fruit_type"],
        "model_c_fruit": detection.get("model_c_fruit_type"),
        "model_c_matches_final": detection.get("model_c_matches_final"),
        "raw_model_c_ripeness": detection.get("model_c_raw_ripeness"),
        "gated_model_c_ripeness": detection.get("model_c_ripeness"),
        "baseline_ripeness": raw_fusion["ripeness"],
        "gated_ripeness": gated_fusion["ripeness"],
        "baseline_correct": raw_fusion["ripeness"] == expected,
        "gated_correct": gated_fusion["ripeness"] == expected,
    }


def main():
    samples = load_samples()
    rows = []

    for index, sample in enumerate(samples, start=1):
        result = evaluate_sample(sample)
        rows.append({
            "image": str(sample["image"]),
            "expected_fruit": sample["expected_fruit"],
            "expected_ripeness": sample["expected_ripeness"],
            **result,
        })
        print(f"Model-gate regression: {index}/{len(samples)}", flush=True)

    fieldnames = sorted({key for row in rows for key in row})
    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    evaluated = [row for row in rows if row["status"] == "evaluated"]
    changed = [
        row for row in evaluated
        if row["baseline_ripeness"] != row["gated_ripeness"]
    ]
    baseline_correct = sum(bool(row["baseline_correct"]) for row in evaluated)
    gated_correct = sum(bool(row["gated_correct"]) for row in evaluated)
    summary = {
        "samples": len(samples),
        "evaluated_after_correct_fruit_detection": len(evaluated),
        "model_c_mismatch_cases": sum(
            row.get("model_c_matches_final") is False for row in evaluated
        ),
        "final_ripeness_changed_cases": len(changed),
        "baseline_correct": baseline_correct,
        "gated_correct": gated_correct,
        "baseline_accuracy": baseline_correct / len(evaluated),
        "gated_accuracy": gated_correct / len(evaluated),
    }
    OUTPUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
