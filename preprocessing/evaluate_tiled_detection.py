"""Regression-test full-frame detection against full-frame plus tiled detection."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2

from fruit_ripeness_object_detection.fruit_detection import (
    detect_with_model_a,
    detect_with_model_c,
    detect_with_model_d,
    fuse_detections,
)
from preprocessing import preprocess_fruit_image
from preprocessing.tiled_preprocessing import (
    create_overlapping_tiles,
    map_tile_detections_to_standard_image,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVALUATION_DIRECTORY = PROJECT_ROOT / "preprocessing_evaluation"
PRIMARY_CSV = EVALUATION_DIRECTORY / "dataset_tuning_summary.csv"
ARCHIVE_CSV = EVALUATION_DIRECTORY / "archive_tuning_summary.csv"
EXTERNAL_DIRECTORY = PROJECT_ROOT / "external_system_evaluation"
OUTPUT_CSV = EVALUATION_DIRECTORY / "tiled_detection_regression.csv"
OUTPUT_SUMMARY = EVALUATION_DIRECTORY / "tiled_detection_regression.json"

EXTERNAL_PRESENCE_LABELS = {
    "01_fruit_bowl.jpg": {"Apple", "Banana", "Orange", "Pear"},
    "02_bowl_of_fruit.jpg": {"Apple", "Orange"},
    "03_culinary_fruits.jpg": {
        "Apple", "Banana", "Grape", "Mango", "Melon", "Orange", "Pear",
    },
    "04_fruits_in_basket.jpg": {"Apple", "Banana", "Grape", "Orange"},
}


def evenly_spaced_sample(items, sample_size):
    if len(items) <= sample_size:
        return list(items)

    return [
        items[round(index * (len(items) - 1) / (sample_size - 1))]
        for index in range(sample_size)
    ]


def load_single_fruit_regression_sample():
    grouped = defaultdict(dict)

    with PRIMARY_CSV.open(encoding="utf-8-sig", newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            key = (row["folder"], row["image"])
            grouped["primary"][key] = {
                "image": Path(row["image"]),
                "expected_fruit": row["expected_fruit"],
                "folder": row["folder"],
            }

    with ARCHIVE_CSV.open(encoding="utf-8-sig", newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            key = (row["folder"], row["image"])
            grouped["archive"][key] = {
                "image": Path(row["image"]),
                "expected_fruit": row["expected_fruit"],
                "folder": row["folder"],
            }

    primary_by_folder = defaultdict(list)
    for sample in grouped["primary"].values():
        primary_by_folder[sample["folder"]].append(sample)

    samples = []
    for folder in sorted(primary_by_folder):
        folder_samples = sorted(
            primary_by_folder[folder],
            key=lambda item: str(item["image"]),
        )
        samples.extend(evenly_spaced_sample(folder_samples, 10))

    samples.extend(
        sorted(
            grouped["archive"].values(),
            key=lambda item: (item["folder"], str(item["image"])),
        )
    )
    return samples


def run_full_frame(preprocessing_results, confidence_threshold=0.30):
    image = preprocessing_results["classification_image"]
    detections_a = detect_with_model_a(image, confidence_threshold)
    detections_c = detect_with_model_c(image, confidence_threshold)
    detections_d = detect_with_model_d(image, confidence_threshold)
    return detections_a, detections_c, detections_d


def run_tiled_pass(preprocessing_results, confidence_threshold=0.30):
    tiles = create_overlapping_tiles(
        preprocessing_results["source_image_full_resolution"],
        tile_size=(1280, 1280),
        overlap_ratio=0.20,
        output_size=preprocessing_results["output_size"],
    )
    tiled_a = []
    tiled_c = []
    tiled_d = []

    for tile in tiles:
        tile_image = tile["classification_image"]
        model_detections = (
            (tiled_a, detect_with_model_a(tile_image, confidence_threshold)),
            (tiled_c, detect_with_model_c(tile_image, confidence_threshold)),
            (tiled_d, detect_with_model_d(tile_image, confidence_threshold)),
        )

        for destination, detections in model_detections:
            destination.extend(
                map_tile_detections_to_standard_image(
                    detections,
                    tile,
                    preprocessing_results["resize_scale"],
                    preprocessing_results["resize_padding"],
                    preprocessing_results["output_size"],
                )
            )

    return tiled_a, tiled_c, tiled_d, len(tiles)


def detect_image(image_path, confidence_threshold=0.30):
    preprocessing_results = preprocess_fruit_image(image_path)
    full_a, full_c, full_d = run_full_frame(
        preprocessing_results,
        confidence_threshold,
    )
    baseline = fuse_detections(full_a, full_c, full_d)
    tiled_a, tiled_c, tiled_d, tile_count = run_tiled_pass(
        preprocessing_results,
        confidence_threshold,
    )
    candidate = fuse_detections(
        full_a,
        full_c,
        full_d + tiled_d,
    )
    return baseline, candidate, tile_count


def top_prediction(detections):
    if not detections:
        return "", 0.0

    detection = max(detections, key=lambda item: item["confidence"])
    return detection["fruit_type"], detection["confidence"]


def evaluate_single_fruit_samples(samples, confidence_threshold):
    rows = []

    for index, sample in enumerate(samples, start=1):
        baseline, candidate, tile_count = detect_image(
            sample["image"],
            confidence_threshold,
        )
        baseline_class, baseline_confidence = top_prediction(baseline)
        candidate_class, candidate_confidence = top_prediction(candidate)
        expected = sample["expected_fruit"]
        rows.append({
            "evaluation": "single_fruit",
            "image": str(sample["image"]),
            "expected": expected,
            "baseline_prediction": baseline_class,
            "baseline_confidence": baseline_confidence,
            "baseline_correct": baseline_class.casefold() == expected.casefold(),
            "candidate_prediction": candidate_class,
            "candidate_confidence": candidate_confidence,
            "candidate_correct": candidate_class.casefold() == expected.casefold(),
            "tile_count": tile_count,
        })
        print(f"Single-fruit regression: {index}/{len(samples)}", flush=True)

    return rows


def evaluate_external_images(confidence_threshold):
    rows = []

    for image_name, expected_classes in EXTERNAL_PRESENCE_LABELS.items():
        image_path = EXTERNAL_DIRECTORY / image_name
        baseline, candidate, tile_count = detect_image(
            image_path,
            confidence_threshold,
        )
        baseline_classes = {item["fruit_type"] for item in baseline}
        candidate_classes = {item["fruit_type"] for item in candidate}
        rows.append({
            "evaluation": "external_presence",
            "image": str(image_path),
            "expected": json.dumps(sorted(expected_classes)),
            "baseline_prediction": json.dumps(sorted(baseline_classes)),
            "baseline_confidence": "",
            "baseline_correct": len(expected_classes & baseline_classes),
            "candidate_prediction": json.dumps(sorted(candidate_classes)),
            "candidate_confidence": "",
            "candidate_correct": len(expected_classes & candidate_classes),
            "tile_count": tile_count,
        })

    return rows


def create_summary(rows):
    single_rows = [row for row in rows if row["evaluation"] == "single_fruit"]
    external_rows = [
        row for row in rows if row["evaluation"] == "external_presence"
    ]
    baseline_correct = sum(bool(row["baseline_correct"]) for row in single_rows)
    candidate_correct = sum(bool(row["candidate_correct"]) for row in single_rows)
    external_expected = sum(
        len(json.loads(row["expected"])) for row in external_rows
    )
    baseline_external_hits = sum(
        int(row["baseline_correct"]) for row in external_rows
    )
    candidate_external_hits = sum(
        int(row["candidate_correct"]) for row in external_rows
    )
    return {
        "single_fruit_images": len(single_rows),
        "baseline_correct": baseline_correct,
        "candidate_correct": candidate_correct,
        "baseline_accuracy": baseline_correct / len(single_rows),
        "candidate_accuracy": candidate_correct / len(single_rows),
        "external_expected_class_instances": external_expected,
        "baseline_external_class_hits": baseline_external_hits,
        "candidate_external_class_hits": candidate_external_hits,
        "baseline_external_presence_recall": (
            baseline_external_hits / external_expected
        ),
        "candidate_external_presence_recall": (
            candidate_external_hits / external_expected
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--confidence-threshold", type=float, default=0.30)
    parser.add_argument("--limit", type=int)
    arguments = parser.parse_args()
    samples = load_single_fruit_regression_sample()

    if arguments.limit is not None:
        samples = samples[:arguments.limit]

    rows = evaluate_single_fruit_samples(
        samples,
        arguments.confidence_threshold,
    )
    rows.extend(evaluate_external_images(arguments.confidence_threshold))
    summary = create_summary(rows)

    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    OUTPUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
