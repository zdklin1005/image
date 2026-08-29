"""Strict external and perturbation testing for the integrated fruit system.

Production modules are imported read-only.  Generated variants, annotations,
CSV data, and the JSON summary are written under strict_regression_results.
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from fruit_ripeness_object_detection.evaluate_external_system import (
    evaluate_image,
)
from fruit_ripeness_object_detection.fruit_detection import (
    assess_detection_quality,
    detect_with_model_a,
    detect_with_model_c,
    detect_with_model_d,
    draw_final_detections,
    fuse_detections,
)
from preprocessing import preprocess_fruit_image


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT_ROOT / "strict_regression_results"
EXTERNAL = RESULTS / "external_inputs"
VARIANTS = RESULTS / "robustness_variants"
ANNOTATED = RESULTS / "external_annotated"
CSV_PATH = RESULTS / "external_robustness.csv"
JSON_PATH = RESULTS / "external_robustness_summary.json"
FULL_SYSTEM_CSV = RESULTS / "external_full_system.csv"


CASES = [
    (PROJECT_ROOT / "external_system_evaluation/01_fruit_bowl.jpg",
     {"Apple", "Banana", "Orange", "Pear"}, "existing_external"),
    (PROJECT_ROOT / "external_system_evaluation/02_bowl_of_fruit.jpg",
     {"Apple", "Orange"}, "existing_external"),
    (PROJECT_ROOT / "external_system_evaluation/03_culinary_fruits.jpg",
     {"Apple", "Banana", "Grape", "Mango", "Melon", "Orange", "Pear",
      "Pineapple", "Watermelon"}, "existing_external"),
    (PROJECT_ROOT / "external_system_evaluation/04_fruits_in_basket.jpg",
     {"Apple", "Banana", "Grape", "Orange"}, "existing_external"),
    (EXTERNAL / "01_mango_single.jpg", {"Mango"}, "new_internet"),
    (EXTERNAL / "02_pear_single.jpg", {"Pear"}, "new_internet"),
    (EXTERNAL / "03_peach_single.jpg", {"Peach"}, "new_internet"),
    (EXTERNAL / "04_grape_single.jpg", {"Grape"}, "new_internet"),
    (EXTERNAL / "06_pineapple_single.jpg", {"Pineapple"}, "new_internet"),
    (EXTERNAL / "08_mixed_natural_light.jpg",
     {"Apple", "Banana", "Grape", "Orange", "Pear"}, "new_internet"),
    (PROJECT_ROOT / "img_5874-1.png", {"Mango"}, "user_supplied"),
    (Path("C:/Users/Kai/Downloads/WhatsApp Image 2026-08-28 at 8.27.10 PM.jpeg"),
     {"Pineapple"}, "user_supplied"),
    (Path("C:/Users/Kai/Downloads/WhatsApp Image 2026-08-28 at 8.27.26 PM.jpeg"),
     {"Apple", "Banana", "Mango", "Orange", "Pineapple"}, "user_supplied"),
]


def create_variant(image, variant):
    if variant == "original":
        return image.copy()
    if variant == "dark_45_percent":
        return cv2.convertScaleAbs(image, alpha=0.45, beta=0)
    if variant == "bright_140_percent":
        return cv2.convertScaleAbs(image, alpha=1.40, beta=20)
    if variant == "gaussian_blur_9":
        return cv2.GaussianBlur(image, (9, 9), 0)
    if variant == "salt_pepper_1_percent":
        noisy = image.copy()
        rng = np.random.default_rng(2133)
        pixel_count = image.shape[0] * image.shape[1]
        count = max(1, int(pixel_count * 0.005))
        for value in (0, 255):
            ys = rng.integers(0, image.shape[0], count)
            xs = rng.integers(0, image.shape[1], count)
            noisy[ys, xs] = value
        return noisy
    if variant == "rotate_15_degrees":
        height, width = image.shape[:2]
        transform = cv2.getRotationMatrix2D((width / 2, height / 2), 15, 1.0)
        return cv2.warpAffine(
            image,
            transform,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
    raise ValueError(f"Unknown variant: {variant}")


def run_detector(preprocessing):
    image = preprocessing["classification_image"]
    detections = fuse_detections(
        detect_with_model_a(image, 0.30),
        detect_with_model_c(image, 0.30),
        detect_with_model_d(image, 0.30),
        iou_threshold=0.30,
    )
    return assess_detection_quality(
        detections,
        image.shape,
        valid_content_bbox=preprocessing["valid_content_bbox"],
        retain_rejected=True,
    )


def main():
    VARIANTS.mkdir(parents=True, exist_ok=True)
    ANNOTATED.mkdir(parents=True, exist_ok=True)
    variants = (
        "original",
        "dark_45_percent",
        "bright_140_percent",
        "gaussian_blur_9",
        "salt_pepper_1_percent",
        "rotate_15_degrees",
    )
    rows = []
    full_system_rows = []
    missing = []

    existing_cases = [case for case in CASES if case[0].exists()]
    for path, _, _ in CASES:
        if not path.exists():
            missing.append(str(path))

    for case_index, (path, expected, source_group) in enumerate(existing_cases, 1):
        source = cv2.imread(str(path))
        if source is None:
            missing.append(str(path))
            continue
        case_id = f"{case_index:02}_{path.stem[:45]}"
        print(f"[{case_index:02}/{len(existing_cases):02}] {path.name}")

        # Run every original image through the complete integrated path,
        # including ROI segmentation, ripeness, and blemish analysis.
        try:
            preprocessing, detections, image_rows, annotated = evaluate_image(path)
            cv2.imwrite(str(ANNOTATED / f"{case_id}.jpg"), annotated)
            for result in image_rows:
                full_system_rows.append({
                    "case_id": case_id,
                    "source_group": source_group,
                    "source_image": str(path),
                    **result,
                })
        except Exception as error:
            full_system_rows.append({
                "case_id": case_id,
                "source_group": source_group,
                "source_image": str(path),
                "pipeline_error": f"{type(error).__name__}: {error}",
            })

        for variant in variants:
            variant_image = create_variant(source, variant)
            variant_path = VARIANTS / f"{case_id}__{variant}.jpg"
            cv2.imwrite(str(variant_path), variant_image)
            preprocessing = preprocess_fruit_image(variant_path)
            detections = run_detector(preprocessing)
            usable = [
                detection
                for detection in detections
                if detection.get("reliability_status") != "Rejected"
            ]
            predicted = [detection["fruit_type"] for detection in usable]
            predicted_set = set(predicted)
            hits = expected & predicted_set
            false_classes = predicted_set - expected
            rows.append({
                "case_id": case_id,
                "source_group": source_group,
                "source_image": str(path),
                "variant": variant,
                "expected_classes": "; ".join(sorted(expected)),
                "predicted_classes": "; ".join(predicted),
                "expected_class_count": len(expected),
                "hit_count": len(hits),
                "presence_recall": len(hits) / len(expected),
                "false_class_count": len(false_classes),
                "false_classes": "; ".join(sorted(false_classes)),
                "detection_count": len(usable),
                "rejected_box_count": len(detections) - len(usable),
                "duplicate_box_count": max(0, len(predicted) - len(predicted_set)),
                "preprocessing_suitability": preprocessing.get(
                    "preprocessing_suitability"
                ),
                "blur_status": preprocessing.get("blur_status"),
                "exposure_status": preprocessing.get("exposure_status"),
                "contrast_status": preprocessing.get("contrast_status"),
                "blur_score": preprocessing.get("blur_score"),
            })

    def write_csv(path, data):
        fields = []
        for item in data:
            for field in item:
                if field not in fields:
                    fields.append(field)
        with path.open("w", encoding="utf-8-sig", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(data)

    write_csv(CSV_PATH, rows)
    write_csv(FULL_SYSTEM_CSV, full_system_rows)

    by_variant = defaultdict(list)
    for row in rows:
        by_variant[row["variant"]].append(row)

    def aggregate(group):
        expected = sum(row["expected_class_count"] for row in group)
        hits = sum(row["hit_count"] for row in group)
        return {
            "images": len(group),
            "expected_class_instances": expected,
            "detected_expected_classes": hits,
            "presence_recall": hits / expected if expected else 0.0,
            "images_with_no_detection": sum(
                row["detection_count"] == 0 for row in group
            ),
            "false_class_predictions": sum(
                row["false_class_count"] for row in group
            ),
            "duplicate_boxes": sum(row["duplicate_box_count"] for row in group),
            "preprocessing_not_acceptable": sum(
                row["preprocessing_suitability"] != "Acceptable"
                for row in group
            ),
        }

    original_rows = by_variant["original"]
    roi_errors = sum(bool(row.get("roi_error")) for row in full_system_rows)
    pipeline_errors = sum(bool(row.get("pipeline_error")) for row in full_system_rows)
    summary = {
        "scope": {
            "base_images": len(existing_cases),
            "variants_per_image": len(variants),
            "detector_runs": len(rows),
            "missing_inputs": missing,
        },
        "original_external_images": aggregate(original_rows),
        "by_variant": {
            name: aggregate(group)
            for name, group in sorted(by_variant.items())
        },
        "full_system_original_runs": {
            "result_rows": len(full_system_rows),
            "roi_or_downstream_errors": roi_errors,
            "pipeline_errors": pipeline_errors,
        },
    }
    JSON_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Detector CSV: {CSV_PATH}")
    print(f"Full-system CSV: {FULL_SYSTEM_CSV}")
    print(f"Summary: {JSON_PATH}")


if __name__ == "__main__":
    main()
