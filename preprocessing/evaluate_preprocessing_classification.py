"""Reproduce the classification-based preprocessing tuning experiments.

This evaluation utility is intentionally separate from the runtime pipeline.
It compares five preprocessing variants, runs the existing YOLO detector, and
writes one auditable CSV record for every image/variant combination.

Default experiments:

* primary: 6 supported folders x 5 image categories x 3 images = 90 images
* archive: 8 supported folders x 10 images = 80 images

Each selected image is evaluated using five preprocessing variants, producing
450 primary rows and 400 archive rows with the default settings.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

try:
    from .preprocessing import preprocess_fruit_image
except ImportError:
    from preprocessing import preprocess_fruit_image


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSIGNMENT_ROOT = PROJECT_ROOT.parent

IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
}

PRIMARY_FOLDERS = {
    "freshapples": ("Apple", "Fresh"),
    "freshbanana": ("Banana", "Fresh"),
    "freshoranges": ("Orange", "Fresh"),
    "rottenapples": ("Apple", "Rotten"),
    "rottenbanana": ("Banana", "Rotten"),
    "rottenoranges": ("Orange", "Rotten"),
}

PRIMARY_CATEGORIES = (
    "original",
    "rotated",
    "vertical",
    "translation",
    "saltandpepper",
)

ARCHIVE_FOLDERS = {
    "Ripe Apple": ("Apple", "Ripe"),
    "Ripe Banana": ("Banana", "Ripe"),
    "Ripe Grape": ("Grape", "Ripe"),
    "Ripe Mango": ("Mango", "Ripe"),
    "Unripe Apple": ("Apple", "Unripe"),
    "Unripe Banana": ("Banana", "Unripe"),
    "Unripe Grape": ("Grape", "Unripe"),
    "Unripe Mango": ("Mango", "Unripe"),
}

VARIANT_ORDER = (
    "resized_only",
    "median3_bilateral",
    "median5_bilateral",
    "median3_clahe",
    "median5_clahe",
)


def find_images(folder):
    """Return supported image files directly inside a folder."""
    folder = Path(folder)

    if not folder.is_dir():
        raise FileNotFoundError(f"Dataset folder does not exist: {folder}")

    return sorted(
        (
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda path: path.name.casefold(),
    )


def evenly_spaced_sample(items, sample_size):
    """Select deterministic, evenly distributed items from a sorted list."""
    if sample_size <= 0:
        raise ValueError("sample_size must be greater than zero.")

    items = list(items)

    if len(items) < sample_size:
        raise ValueError(
            f"Requested {sample_size} images, but only {len(items)} are available."
        )

    if sample_size == 1:
        return [items[len(items) // 2]]

    indices = [
        round(index * (len(items) - 1) / (sample_size - 1))
        for index in range(sample_size)
    ]

    return [items[index] for index in indices]


def classify_primary_category(image_path):
    """Classify a primary-dataset filename into its transformation group."""
    filename = image_path.name.casefold()

    if filename.startswith("rotated_"):
        return "rotated"
    if filename.startswith("vertical_"):
        return "vertical"
    if filename.startswith("translation_"):
        return "translation"
    if filename.startswith("saltandpepper_"):
        return "saltandpepper"

    return "original"


def select_primary_images(dataset_root, images_per_category=3):
    """Select a reproducible stratified sample from the primary dataset."""
    dataset_root = Path(dataset_root)
    selected = []

    for folder_name, labels in PRIMARY_FOLDERS.items():
        folder = dataset_root / folder_name
        images = find_images(folder)

        grouped_images = {
            category: [] for category in PRIMARY_CATEGORIES
        }

        for image_path in images:
            category = classify_primary_category(image_path)
            grouped_images[category].append(image_path)

        for category in PRIMARY_CATEGORIES:
            sample = evenly_spaced_sample(
                grouped_images[category],
                images_per_category,
            )

            for image_path in sample:
                selected.append(
                    {
                        "folder": folder_name,
                        "category": category,
                        "image": image_path,
                        "expected_fruit": labels[0],
                        "freshness_label": labels[1],
                    }
                )

    return selected


def select_archive_images(archive_root, images_per_folder=10):
    """Select a reproducible sample from supported archive folders."""
    archive_root = Path(archive_root)
    selected = []

    for folder_name, labels in ARCHIVE_FOLDERS.items():
        folder = archive_root / folder_name
        source_groups = defaultdict(list)

        # Roboflow-style filenames contain a source-image identifier before
        # ".rf.". Grouping on that identifier prevents augmented copies of the
        # same source image from being sampled more than once.
        for image_path in sorted(folder.glob("*.jpg")):
            source_id = image_path.name.split(".rf.", 1)[0]
            source_groups[source_id].append(image_path)

        source_representatives = [
            source_groups[source_id][0]
            for source_id in sorted(source_groups)
        ]

        sample = evenly_spaced_sample(
            source_representatives,
            images_per_folder,
        )

        for image_path in sample:
            selected.append(
                {
                    "folder": folder_name,
                    "image": image_path,
                    "expected_fruit": labels[0],
                    "expected_ripeness": labels[1],
                }
            )

    return selected


def create_preprocessing_variants(image_path):
    """Create the five image variants used by the tuning experiment."""
    median3_results = preprocess_fruit_image(
        image_path,
        median_kernel=3,
    )
    median5_results = preprocess_fruit_image(
        image_path,
        median_kernel=5,
    )

    variants = {
        "resized_only": median3_results["original_image"],
        "median3_bilateral": median3_results["bilateral_image"],
        "median5_bilateral": median5_results["bilateral_image"],
        "median3_clahe": median3_results["analysis_image"],
        "median5_clahe": median5_results["analysis_image"],
    }

    return variants, median3_results["valid_content_bbox"]


def detect_variants(
    variants,
    confidence_threshold,
    valid_content_bbox,
):
    """Evaluate every variant with the current integrated model gates."""
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

    predictions = {}

    for variant_name in VARIANT_ORDER:
        image = variants[variant_name]
        detections_a = detect_with_model_a(
            image,
            confidence_threshold,
        )
        detections_c = detect_with_model_c(
            image,
            confidence_threshold,
        )
        detections_d = detect_with_model_d(
            image,
            confidence_threshold,
        )
        detections = fuse_detections(
            detections_a,
            detections_c,
            detections_d,
            iou_threshold=0.30,
        )
        detections = assess_detection_quality(
            detections,
            image.shape,
            valid_content_bbox=valid_content_bbox,
            retain_rejected=False,
        )

        if not detections:
            predictions[variant_name] = {
                "predicted_fruit": "",
                "predicted_ripeness": "",
                "confidence": 0.0,
                "detected": False,
            }
            continue

        detection = max(
            detections,
            key=lambda item: item["confidence"],
        )
        x1, y1, x2, y2 = detection["bounding_box"]
        fruit_roi = image[y1:y2, x1:x2].copy()
        predicted_ripeness = ""

        if (
            detection.get("ripeness_supported", False)
            and fruit_roi.size > 0
        ):
            model_b_result = classify_with_model_b(
                fruit_roi,
                detection["fruit_type"],
            )
            model_e_result = classify_with_model_e(
                fruit_roi,
            )
            ripeness_result = fuse_ripeness(
                model_b_result,
                detection.get("model_c_ripeness"),
                detection.get("confidence_c"),
                model_e_result,
            )
            predicted_ripeness = ripeness_result["ripeness"]

        predictions[variant_name] = {
            "predicted_fruit": detection["fruit_type"],
            "predicted_ripeness": predicted_ripeness,
            "confidence": detection["confidence"],
            "detected": True,
        }

    return predictions


def load_primary_selection(result_csv):
    """Recover the exact primary image sample recorded in an existing CSV."""
    selections = []
    seen_images = set()

    with Path(result_csv).open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:
        for row in csv.DictReader(csv_file):
            image_path = Path(row["image"])
            image_key = str(image_path).casefold()

            if image_key in seen_images:
                continue

            if not image_path.is_file():
                raise FileNotFoundError(
                    f"Recorded evaluation image no longer exists: {image_path}"
                )

            seen_images.add(image_key)
            selections.append(
                {
                    "folder": row["folder"],
                    "category": row["category"],
                    "image": image_path,
                    "expected_fruit": row["expected_fruit"],
                    "freshness_label": row["freshness_label"],
                }
            )

    return selections


def load_archive_selection(result_csv):
    """Recover the exact archive image sample recorded in an existing CSV."""
    selections = []
    seen_images = set()

    with Path(result_csv).open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:
        for row in csv.DictReader(csv_file):
            image_path = Path(row["image"])
            image_key = str(image_path).casefold()

            if image_key in seen_images:
                continue

            if not image_path.is_file():
                raise FileNotFoundError(
                    f"Recorded evaluation image no longer exists: {image_path}"
                )

            seen_images.add(image_key)
            selections.append(
                {
                    "folder": row["folder"],
                    "image": image_path,
                    "expected_fruit": row["expected_fruit"],
                    "expected_ripeness": row["expected_ripeness"],
                }
            )

    return selections


def evaluate_primary(
    dataset_root,
    images_per_category,
    confidence_threshold,
    selected_images=None,
):
    """Evaluate all preprocessing variants on the primary sample."""
    rows = []
    if selected_images is None:
        selected_images = select_primary_images(
            dataset_root,
            images_per_category,
        )

    for image_number, sample in enumerate(selected_images, start=1):
        print(
            f"Primary image {image_number}/{len(selected_images)}: "
            f"{sample['image'].name}"
        )

        variants, valid_content_bbox = create_preprocessing_variants(
            sample["image"]
        )
        predictions = detect_variants(
            variants,
            confidence_threshold,
            valid_content_bbox,
        )

        for variant_name in VARIANT_ORDER:
            prediction = predictions[variant_name]
            fruit_correct = (
                prediction["predicted_fruit"].casefold()
                == sample["expected_fruit"].casefold()
            )

            if sample["freshness_label"] == "Rotten":
                rotten_correct = (
                    prediction["predicted_ripeness"].casefold()
                    == "rotten"
                )
            else:
                # The primary dataset's Fresh label does not map uniquely to
                # the model's Ripe, Unripe and Overripe classes.
                rotten_correct = ""

            rows.append(
                {
                    "folder": sample["folder"],
                    "category": sample["category"],
                    "image": str(sample["image"].resolve()),
                    "variant": variant_name,
                    "expected_fruit": sample["expected_fruit"],
                    "freshness_label": sample["freshness_label"],
                    **prediction,
                    "fruit_correct": fruit_correct,
                    "rotten_correct": rotten_correct,
                }
            )

    return rows


def evaluate_archive(
    archive_root,
    images_per_folder,
    confidence_threshold,
    selected_images=None,
):
    """Evaluate all preprocessing variants on the archive sample."""
    rows = []
    if selected_images is None:
        selected_images = select_archive_images(
            archive_root,
            images_per_folder,
        )

    for image_number, sample in enumerate(selected_images, start=1):
        print(
            f"Archive image {image_number}/{len(selected_images)}: "
            f"{sample['image'].name}"
        )

        variants, valid_content_bbox = create_preprocessing_variants(
            sample["image"]
        )
        predictions = detect_variants(
            variants,
            confidence_threshold,
            valid_content_bbox,
        )

        for variant_name in VARIANT_ORDER:
            prediction = predictions[variant_name]
            fruit_correct = (
                prediction["predicted_fruit"].casefold()
                == sample["expected_fruit"].casefold()
            )
            ripeness_correct = (
                prediction["predicted_ripeness"].casefold()
                == sample["expected_ripeness"].casefold()
            )

            rows.append(
                {
                    "folder": sample["folder"],
                    "image": str(sample["image"].resolve()),
                    "variant": variant_name,
                    "expected_fruit": sample["expected_fruit"],
                    "expected_ripeness": sample["expected_ripeness"],
                    **prediction,
                    "fruit_correct": fruit_correct,
                    "ripeness_correct": ripeness_correct,
                    "exact_correct": fruit_correct and ripeness_correct,
                }
            )

    return rows


def percentage(rows, field_name):
    """Return the percentage of rows whose Boolean field is true."""
    if not rows:
        return 0.0

    true_count = sum(row[field_name] is True for row in rows)
    return 100.0 * true_count / len(rows)


def print_summary(rows, include_ripeness=False):
    """Print the same percentages used by the assignment tables."""
    print("\nEvaluation summary")
    print("=" * 76)
    print(
        f"{'Variant':24} {'Images':>8} {'Detected':>11} "
        f"{'Fruit correct':>15}",
        end="",
    )

    if include_ripeness:
        print(f" {'Ripeness':>11} {'Exact':>9}")
    else:
        print()

    for variant_name in VARIANT_ORDER:
        variant_rows = [
            row for row in rows if row["variant"] == variant_name
        ]
        line = (
            f"{variant_name:24} {len(variant_rows):8d} "
            f"{percentage(variant_rows, 'detected'):10.2f}% "
            f"{percentage(variant_rows, 'fruit_correct'):14.2f}%"
        )

        if include_ripeness:
            line += (
                f" {percentage(variant_rows, 'ripeness_correct'):10.2f}%"
                f" {percentage(variant_rows, 'exact_correct'):8.2f}%"
            )

        print(line)


def write_csv(rows, output_path, overwrite=False):
    """Write evaluation records without silently replacing prior evidence."""
    output_path = Path(output_path)

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}\n"
            "Use --overwrite to replace it."
        )

    if not rows:
        raise ValueError("No evaluation rows were produced.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCSV saved to: {output_path.resolve()}")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce classification-based preprocessing tuning results."
        )
    )
    parser.add_argument(
        "mode",
        choices=("primary", "archive", "all"),
        help="Dataset evaluation to run.",
    )
    parser.add_argument(
        "--primary-source",
        type=Path,
        default=ASSIGNMENT_ROOT / "dataset" / "Test",
        help="Primary Test dataset folder.",
    )
    parser.add_argument(
        "--archive-source",
        type=Path,
        default=ASSIGNMENT_ROOT / "archive (1)",
        help="Independent archive dataset folder.",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=PROJECT_ROOT / "preprocessing_evaluation",
        help="Folder in which the CSV files will be written.",
    )
    parser.add_argument(
        "--primary-images-per-category",
        type=int,
        default=3,
        help="Images sampled from each primary folder/category combination.",
    )
    parser.add_argument(
        "--archive-images-per-folder",
        type=int,
        default=10,
        help="Images sampled from each supported archive folder.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.40,
        help="Minimum YOLO detection confidence.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing tuning CSV files.",
    )
    parser.add_argument(
        "--resample",
        action="store_true",
        help=(
            "Select a new deterministic sample instead of reusing the exact "
            "image list recorded in an existing output CSV."
        ),
    )

    arguments = parser.parse_args()

    if not 0.0 <= arguments.confidence_threshold <= 1.0:
        parser.error("--confidence-threshold must be between 0 and 1.")

    if arguments.primary_images_per_category <= 0:
        parser.error("--primary-images-per-category must be positive.")

    if arguments.archive_images_per_folder <= 0:
        parser.error("--archive-images-per-folder must be positive.")

    return arguments


def main():
    arguments = parse_arguments()

    if arguments.mode in {"primary", "all"}:
        primary_output = (
            arguments.output_directory / "dataset_tuning_summary.csv"
        )

        if primary_output.exists() and not arguments.overwrite:
            raise FileExistsError(
                f"Output already exists: {primary_output}\n"
                "Use --overwrite to reproduce it."
            )

        primary_selection = None

        if primary_output.exists() and not arguments.resample:
            primary_selection = load_primary_selection(primary_output)
            print(
                "Reusing the exact primary image selection recorded in "
                f"{primary_output.name}."
            )

        primary_rows = evaluate_primary(
            arguments.primary_source,
            arguments.primary_images_per_category,
            arguments.confidence_threshold,
            selected_images=primary_selection,
        )
        print_summary(primary_rows)
        write_csv(
            primary_rows,
            primary_output,
            overwrite=arguments.overwrite,
        )

    if arguments.mode in {"archive", "all"}:
        archive_output = (
            arguments.output_directory / "archive_tuning_summary.csv"
        )

        if archive_output.exists() and not arguments.overwrite:
            raise FileExistsError(
                f"Output already exists: {archive_output}\n"
                "Use --overwrite to reproduce it."
            )

        archive_selection = None

        if archive_output.exists() and not arguments.resample:
            archive_selection = load_archive_selection(archive_output)
            print(
                "Reusing the exact archive image selection recorded in "
                f"{archive_output.name}."
            )

        archive_rows = evaluate_archive(
            arguments.archive_source,
            arguments.archive_images_per_folder,
            arguments.confidence_threshold,
            selected_images=archive_selection,
        )
        print_summary(archive_rows, include_ripeness=True)
        write_csv(
            archive_rows,
            archive_output,
            overwrite=arguments.overwrite,
        )


if __name__ == "__main__":
    main()
