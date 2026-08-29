"""Strict, repeatable regression tests for the integrated fruit detector.

This file is deliberately separate from the production pipeline.  It compares
the current detector policy with a diagnostic configuration in which Model A
may contribute all of its trained fruit classes.  It never changes model files
or production settings.
"""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from fruit_ripeness_object_detection.fruit_detection import (
    MODEL_A_EXTENSION_FRUITS,
    assess_detection_quality,
    detect_with_model_a,
    detect_with_model_c,
    detect_with_model_d,
    fuse_detections,
)
from preprocessing import preprocess_fruit_image
from preprocessing.evaluate_tiled_detection import (
    load_single_fruit_regression_sample,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIRECTORY = PROJECT_ROOT / "strict_regression_results"
OUTPUT_CSV = OUTPUT_DIRECTORY / "unrestricted_vs_restricted_model_a.csv"
OUTPUT_JSON = OUTPUT_DIRECTORY / "unrestricted_vs_restricted_model_a_summary.json"


def _evaluate_configuration(preprocessing, allow_all_model_a_classes):
    image = preprocessing["classification_image"]
    allowed_fruits = None if allow_all_model_a_classes else MODEL_A_EXTENSION_FRUITS

    detections_a = detect_with_model_a(
        image,
        0.30,
        allowed_fruits=allowed_fruits,
    )
    detections_c = detect_with_model_c(image, 0.30)
    detections_d = detect_with_model_d(image, 0.30)
    fused = fuse_detections(
        detections_a,
        detections_c,
        detections_d,
        iou_threshold=0.30,
    )
    assessed = assess_detection_quality(
        fused,
        image.shape,
        valid_content_bbox=preprocessing["valid_content_bbox"],
        retain_rejected=True,
    )
    usable = [
        detection
        for detection in assessed
        if detection.get("reliability_status") != "Rejected"
    ]
    usable.sort(key=lambda item: item["confidence"], reverse=True)
    return usable, {
        "model_a": len(detections_a),
        "model_c": len(detections_c),
        "model_d": len(detections_d),
    }


def _configuration_fields(prefix, detections, model_counts, expected_fruit):
    predicted = [detection["fruit_type"] for detection in detections]
    expected_matches = [
        detection for detection in detections
        if detection["fruit_type"].casefold() == expected_fruit.casefold()
    ]
    top = detections[0] if detections else None
    false_positive_count = sum(
        fruit.casefold() != expected_fruit.casefold()
        for fruit in predicted
    )

    return {
        f"{prefix}_top_prediction": top["fruit_type"] if top else "No detection",
        f"{prefix}_top_confidence": top["confidence"] if top else None,
        f"{prefix}_top1_correct": bool(
            top and top["fruit_type"].casefold() == expected_fruit.casefold()
        ),
        f"{prefix}_any_hit": bool(expected_matches),
        f"{prefix}_detection_count": len(detections),
        f"{prefix}_expected_box_count": len(expected_matches),
        f"{prefix}_extra_expected_boxes": max(0, len(expected_matches) - 1),
        f"{prefix}_false_positive_count": false_positive_count,
        f"{prefix}_predictions": "; ".join(predicted),
        f"{prefix}_model_a_raw_count": model_counts["model_a"],
        f"{prefix}_model_c_raw_count": model_counts["model_c"],
        f"{prefix}_model_d_raw_count": model_counts["model_d"],
    }


def _summarise(rows, prefix):
    by_class = defaultdict(list)
    for row in rows:
        by_class[row["expected_fruit"]].append(row)

    def aggregate(group):
        count = len(group)
        return {
            "images": count,
            "top1_correct": sum(row[f"{prefix}_top1_correct"] for row in group),
            "top1_accuracy": (
                sum(row[f"{prefix}_top1_correct"] for row in group) / count
                if count else 0.0
            ),
            "any_hit": sum(row[f"{prefix}_any_hit"] for row in group),
            "any_hit_recall": (
                sum(row[f"{prefix}_any_hit"] for row in group) / count
                if count else 0.0
            ),
            "no_detection": sum(
                row[f"{prefix}_detection_count"] == 0 for row in group
            ),
            "false_positive_boxes": sum(
                row[f"{prefix}_false_positive_count"] for row in group
            ),
            "duplicate_expected_boxes": sum(
                row[f"{prefix}_extra_expected_boxes"] for row in group
            ),
        }

    return {
        "overall": aggregate(rows),
        "by_expected_fruit": {
            fruit: aggregate(group)
            for fruit, group in sorted(by_class.items())
        },
    }


def main():
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    samples = load_single_fruit_regression_sample()
    rows = []
    failures = Counter()

    for index, sample in enumerate(samples, start=1):
        image_path = sample["image"]
        expected_fruit = sample["expected_fruit"]
        print(f"[{index:03}/{len(samples):03}] {expected_fruit}: {image_path.name}")

        try:
            preprocessing = preprocess_fruit_image(image_path)
            current, current_counts = _evaluate_configuration(
                preprocessing,
                allow_all_model_a_classes=True,
            )
            diagnostic, diagnostic_counts = _evaluate_configuration(
                preprocessing,
                allow_all_model_a_classes=False,
            )

            row = {
                "image": str(image_path),
                "folder": sample["folder"],
                "expected_fruit": expected_fruit,
                "preprocessing_suitability": preprocessing.get(
                    "preprocessing_suitability"
                ),
                "blur_status": preprocessing.get("blur_status"),
                "exposure_status": preprocessing.get("exposure_status"),
                "contrast_status": preprocessing.get("contrast_status"),
                "blur_score": preprocessing.get("blur_score"),
            }
            row.update(
                _configuration_fields(
                    "current", current, current_counts, expected_fruit
                )
            )
            row.update(
                _configuration_fields(
                    "restricted_model_a",
                    diagnostic,
                    diagnostic_counts,
                    expected_fruit,
                )
            )
            rows.append(row)
        except Exception as error:
            failures[type(error).__name__] += 1
            rows.append({
                "image": str(image_path),
                "folder": sample["folder"],
                "expected_fruit": expected_fruit,
                "error": f"{type(error).__name__}: {error}",
            })

    fieldnames = []
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)

    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    successful_rows = [row for row in rows if "error" not in row]
    summary = {
        "scope": {
            "sample_count": len(samples),
            "successful_images": len(successful_rows),
            "failed_images": len(samples) - len(successful_rows),
            "failure_types": dict(failures),
            "confidence_threshold": 0.30,
            "iou_threshold": 0.30,
        },
        "current_integrated_policy": _summarise(successful_rows, "current"),
        "diagnostic_restricted_model_a_policy": _summarise(
            successful_rows, "restricted_model_a"
        ),
        "comparison": {
            "top1_gained_with_unrestricted_model_a": sum(
                row["current_top1_correct"]
                and (not row["restricted_model_a_top1_correct"])
                for row in successful_rows
            ),
            "top1_lost_with_unrestricted_model_a": sum(
                (not row["current_top1_correct"])
                and row["restricted_model_a_top1_correct"]
                for row in successful_rows
            ),
            "any_hit_gained_with_unrestricted_model_a": sum(
                row["current_any_hit"]
                and (not row["restricted_model_a_any_hit"])
                for row in successful_rows
            ),
            "any_hit_lost_with_unrestricted_model_a": sum(
                (not row["current_any_hit"])
                and row["restricted_model_a_any_hit"]
                for row in successful_rows
            ),
        },
    }
    OUTPUT_JSON.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))
    print(f"CSV: {OUTPUT_CSV}")
    print(f"Summary: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
