import argparse
import csv
import math
import time
from pathlib import Path

import cv2
import numpy as np

try:
    # Used when executed as a package module from the project folder.
    from .preprocessing import preprocess_fruit_image
except ImportError:
    # Used when this file is executed directly.
    from preprocessing import preprocess_fruit_image


PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
}

STAGES = (
    ("original", "original_image"),
    ("median", "median_image"),
    ("bilateral", "bilateral_image"),
    ("clahe_analysis", "analysis_image"),
    ("sharpened_display", "display_image"),
)


def select_source_with_dialog():
    """Open a file-selection dialog when no command-line path is given."""
    from tkinter import Tk, filedialog

    root = Tk()
    root.withdraw()

    selected_path = filedialog.askopenfilename(
        title="Select an image for preprocessing evaluation",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
            ("All files", "*.*"),
        ],
    )

    root.destroy()

    return Path(selected_path) if selected_path else None


def find_images(source_path):
    """Return the supported image files represented by a file or folder."""
    source_path = Path(source_path)

    if source_path.is_file():
        if source_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image type: {source_path.suffix}")

        return [source_path]

    if source_path.is_dir():
        image_paths = sorted(
            path
            for path in source_path.rglob("*")
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        )

        if not image_paths:
            raise FileNotFoundError(
                f"No supported images were found in: {source_path}"
            )

        return image_paths

    raise FileNotFoundError(f"Input path does not exist: {source_path}")


def crop_letterbox_padding(image, padding):
    """Remove letterbox padding before calculating image-quality metrics."""
    left, top, right, bottom = padding
    image_height, image_width = image.shape[:2]

    x_end = image_width - right if right > 0 else image_width
    y_end = image_height - bottom if bottom > 0 else image_height

    cropped_image = image[top:y_end, left:x_end]

    if cropped_image.size == 0:
        raise ValueError("The recorded letterbox padding removed the entire image.")

    return cropped_image


def calculate_entropy(greyscale_image):
    """Calculate Shannon entropy from a greyscale histogram."""
    histogram = cv2.calcHist(
        [greyscale_image],
        [0],
        None,
        [256],
        [0, 256],
    ).ravel()

    probabilities = histogram / histogram.sum()
    probabilities = probabilities[probabilities > 0]

    return float(-np.sum(probabilities * np.log2(probabilities)))


def calculate_stage_metrics(image):
    """Calculate descriptive quality indicators for one processing stage."""
    greyscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    laplacian_variance = float(
        cv2.Laplacian(
            greyscale_image,
            cv2.CV_64F,
            ksize=3,
        ).var()
    )

    edge_image = cv2.Canny(greyscale_image, 100, 200)
    edge_density = float(np.count_nonzero(edge_image) / edge_image.size)

    return {
        "mean_brightness": float(np.mean(greyscale_image)),
        "contrast_standard_deviation": float(np.std(greyscale_image)),
        "entropy": calculate_entropy(greyscale_image),
        "laplacian_variance": laplacian_variance,
        "edge_density": edge_density,
    }


def calculate_psnr(reference_image, evaluated_image):
    """Calculate PSNR when a matching clean reference image is available."""
    if reference_image.shape[:2] != evaluated_image.shape[:2]:
        reference_image = cv2.resize(
            reference_image,
            (evaluated_image.shape[1], evaluated_image.shape[0]),
            interpolation=cv2.INTER_AREA,
        )

    difference = (
        reference_image.astype(np.float64)
        - evaluated_image.astype(np.float64)
    )
    mean_squared_error = float(np.mean(difference ** 2))

    if mean_squared_error == 0:
        return math.inf

    return float(
        20 * math.log10(255.0 / math.sqrt(mean_squared_error))
    )


def validate_results(results, expected_size):
    """Check that the preprocessing contract required by main.py is met."""
    expected_width, expected_height = expected_size
    expected_shape = (expected_height, expected_width, 3)
    checks = []

    for stage_name, result_key in STAGES:
        image = results.get(result_key)

        if not isinstance(image, np.ndarray):
            raise TypeError(f"{result_key} is not a NumPy image array.")

        if image.shape != expected_shape:
            raise ValueError(
                f"{result_key} has shape {image.shape}; "
                f"expected {expected_shape}."
            )

        if image.dtype != np.uint8:
            raise TypeError(
                f"{result_key} uses {image.dtype}; expected uint8."
            )

        checks.append(f"{stage_name}: shape and data type passed")

    if not isinstance(results.get("blur_score"), float):
        raise TypeError("blur_score must be returned as a float.")

    if not isinstance(results.get("is_blurry"), (bool, np.bool_)):
        raise TypeError("is_blurry must be returned as a Boolean value.")

    required_metrics = (
        "mean_brightness",
        "contrast_score",
        "dynamic_range",
        "dark_pixel_ratio",
        "bright_pixel_ratio",
    )

    for metric_name in required_metrics:
        if not isinstance(results.get(metric_name), (float, np.floating)):
            raise TypeError(f"{metric_name} must be returned as a float.")

    checks.append("blur_score: float check passed")
    checks.append("is_blurry: Boolean check passed")
    checks.append("input exposure and contrast metrics: float checks passed")

    return checks


def add_stage_label(image, label):
    """Add a readable title above one stage image."""
    title_height = 44
    labelled_image = cv2.copyMakeBorder(
        image,
        title_height,
        0,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(35, 35, 35),
    )

    cv2.putText(
        labelled_image,
        label,
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return labelled_image


def create_comparison_image(results):
    """Create a labelled contact sheet containing all processing stages."""
    labelled_stages = [
        add_stage_label(results[result_key], stage_name.replace("_", " ").title())
        for stage_name, result_key in STAGES
    ]

    first_row = cv2.hconcat(labelled_stages[:3])

    blank_panel = np.full_like(
        labelled_stages[0],
        255,
    )
    second_row = cv2.hconcat(
        labelled_stages[3:] + [blank_panel]
    )

    return cv2.vconcat([first_row, second_row])


def save_stage_images(results, image_output_directory):
    """Save all individual stages and their comparison contact sheet."""
    image_output_directory.mkdir(parents=True, exist_ok=True)

    for index, (stage_name, result_key) in enumerate(STAGES, start=1):
        output_path = image_output_directory / f"{index:02d}_{stage_name}.png"

        if not cv2.imwrite(str(output_path), results[result_key]):
            raise IOError(f"Unable to save image: {output_path}")

    comparison_path = image_output_directory / "preprocessing_comparison.png"
    comparison_image = create_comparison_image(results)

    if not cv2.imwrite(str(comparison_path), comparison_image):
        raise IOError(f"Unable to save image: {comparison_path}")

    return comparison_path


def write_text_report(
    report_path,
    image_path,
    results,
    processing_time_seconds,
    validation_checks,
    metric_rows,
):
    """Write a readable report for one evaluated image."""
    original_metrics = next(
        row for row in metric_rows if row["stage"] == "original"
    )
    analysis_metrics = next(
        row for row in metric_rows if row["stage"] == "clahe_analysis"
    )
    display_metrics = next(
        row for row in metric_rows if row["stage"] == "sharpened_display"
    )

    contrast_change = (
        analysis_metrics["contrast_standard_deviation"]
        - original_metrics["contrast_standard_deviation"]
    )
    sharpening_change = (
        display_metrics["laplacian_variance"]
        - analysis_metrics["laplacian_variance"]
    )

    with report_path.open("w", encoding="utf-8") as report_file:
        report_file.write("FRUIT IMAGE PREPROCESSING EVALUATION\n")
        report_file.write("=" * 42 + "\n\n")
        report_file.write(f"Input image: {image_path}\n")
        report_file.write(
            f"Output size: {results['output_size'][0]} x "
            f"{results['output_size'][1]} pixels\n"
        )
        report_file.write(
            f"Processing time: {processing_time_seconds:.6f} seconds\n"
        )
        report_file.write(f"Blur score: {results['blur_score']:.4f}\n")
        report_file.write(
            "Blur suitability: "
            + ("Potentially blurry" if results["is_blurry"] else "Acceptable")
            + "\n\n"
        )
        report_file.write(
            f"Input mean brightness: {results['mean_brightness']:.4f}\n"
        )
        report_file.write(
            f"Input contrast score: {results['contrast_score']:.4f}\n"
        )
        report_file.write(
            f"Input dynamic range: {results['dynamic_range']:.4f}\n"
        )
        report_file.write(
            f"Dark pixel ratio: {results['dark_pixel_ratio']:.6f}\n"
        )
        report_file.write(
            f"Bright pixel ratio: {results['bright_pixel_ratio']:.6f}\n"
        )
        report_file.write("Median filter: Applied (fixed 5 x 5)\n")
        report_file.write("Bilateral filter: Applied (fixed parameters)\n\n")

        report_file.write("VALIDATION CHECKS\n")
        report_file.write("-" * 42 + "\n")

        for check in validation_checks:
            report_file.write(f"PASS - {check}\n")

        report_file.write("\nDESCRIPTIVE QUALITY INDICATORS\n")
        report_file.write("-" * 42 + "\n")

        for row in metric_rows:
            report_file.write(f"\nStage: {row['stage']}\n")
            report_file.write(
                f"  Mean brightness: {row['mean_brightness']:.4f}\n"
            )
            report_file.write(
                "  Contrast standard deviation: "
                f"{row['contrast_standard_deviation']:.4f}\n"
            )
            report_file.write(f"  Entropy: {row['entropy']:.4f}\n")
            report_file.write(
                f"  Laplacian variance: {row['laplacian_variance']:.4f}\n"
            )
            report_file.write(
                f"  Edge density: {row['edge_density']:.6f}\n"
            )

            if row.get("psnr_db") is not None:
                psnr_value = row["psnr_db"]
                psnr_text = "infinite" if math.isinf(psnr_value) else f"{psnr_value:.4f}"
                report_file.write(f"  PSNR against reference: {psnr_text} dB\n")

        report_file.write("\nSUMMARY CHANGES\n")
        report_file.write("-" * 42 + "\n")
        report_file.write(
            "CLAHE contrast change from original: "
            f"{contrast_change:+.4f}\n"
        )
        report_file.write(
            "Display sharpness change from analysis image: "
            f"{sharpening_change:+.4f}\n"
        )
        report_file.write(
            "\nThese values are descriptive indicators. Visual inspection and, "
            "when available, PSNR against a matching clean reference should "
            "also be used when discussing preprocessing quality.\n"
        )


def evaluate_image(
    image_path,
    output_directory,
    output_size=(640, 640),
    reference_image=None,
):
    """Evaluate preprocessing for one image and save its evidence."""
    start_time = time.perf_counter()
    results = preprocess_fruit_image(
        image_path=image_path,
        output_size=output_size,
    )
    processing_time_seconds = time.perf_counter() - start_time

    validation_checks = validate_results(results, output_size)
    image_output_directory = output_directory / image_path.stem
    comparison_path = save_stage_images(results, image_output_directory)

    metric_rows = []

    for stage_name, result_key in STAGES:
        content_image = crop_letterbox_padding(
            results[result_key],
            results["resize_padding"],
        )
        metrics = calculate_stage_metrics(content_image)

        row = {
            "input_image": str(image_path),
            "stage": stage_name,
            "processing_time_seconds": processing_time_seconds,
            "blur_score": results["blur_score"],
            "is_blurry": bool(results["is_blurry"]),
            "input_mean_brightness": results["mean_brightness"],
            "input_contrast_score": results["contrast_score"],
            "input_dynamic_range": results["dynamic_range"],
            "dark_pixel_ratio": results["dark_pixel_ratio"],
            "bright_pixel_ratio": results["bright_pixel_ratio"],
            **metrics,
            "psnr_db": None,
        }

        if reference_image is not None:
            row["psnr_db"] = calculate_psnr(
                reference_image,
                content_image,
            )

        metric_rows.append(row)

    report_path = image_output_directory / "evaluation_report.txt"
    write_text_report(
        report_path=report_path,
        image_path=image_path,
        results=results,
        processing_time_seconds=processing_time_seconds,
        validation_checks=validation_checks,
        metric_rows=metric_rows,
    )

    return metric_rows, report_path, comparison_path


def save_combined_csv(metric_rows, csv_path):
    """Save metrics for all evaluated images in one spreadsheet-ready CSV."""
    fieldnames = [
        "input_image",
        "stage",
        "processing_time_seconds",
        "blur_score",
        "is_blurry",
        "input_mean_brightness",
        "input_contrast_score",
        "input_dynamic_range",
        "dark_pixel_ratio",
        "bright_pixel_ratio",
        "mean_brightness",
        "contrast_standard_deviation",
        "entropy",
        "laplacian_variance",
        "edge_density",
        "psnr_db",
    ]

    with csv_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metric_rows)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the fruit-image preprocessing pipeline using one "
            "image or every supported image inside a folder."
        )
    )
    parser.add_argument(
        "source",
        nargs="?",
        help="Path to an input image or a folder of images.",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "preprocessing_evaluation"),
        help="Folder used to save evaluation evidence.",
    )
    parser.add_argument(
        "--reference",
        help=(
            "Optional matching clean reference image for PSNR. "
            "Use this only when evaluating one input image."
        ),
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Standardised output width. Default: 640.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=640,
        help="Standardised output height. Default: 640.",
    )

    return parser.parse_args()


def main():
    arguments = parse_arguments()

    source_path = (
        Path(arguments.source)
        if arguments.source
        else select_source_with_dialog()
    )

    if source_path is None:
        print("No image was selected.")
        return

    image_paths = find_images(source_path)

    if arguments.reference and len(image_paths) != 1:
        raise ValueError(
            "--reference can only be used when evaluating one input image."
        )

    reference_image = None

    if arguments.reference:
        reference_path = Path(arguments.reference)
        reference_image = cv2.imread(str(reference_path))

        if reference_image is None:
            raise FileNotFoundError(
                f"Unable to read reference image: {reference_path}"
            )

    output_directory = Path(arguments.output)
    output_directory.mkdir(parents=True, exist_ok=True)

    all_metric_rows = []

    print(f"Evaluating {len(image_paths)} image(s)...")

    for image_index, image_path in enumerate(image_paths, start=1):
        metric_rows, report_path, comparison_path = evaluate_image(
            image_path=image_path,
            output_directory=output_directory,
            output_size=(arguments.width, arguments.height),
            reference_image=reference_image,
        )
        all_metric_rows.extend(metric_rows)

        print(f"[{image_index}/{len(image_paths)}] {image_path.name}")
        print(f"  Report: {report_path.resolve()}")
        print(f"  Comparison: {comparison_path.resolve()}")

    csv_path = output_directory / "preprocessing_metrics.csv"
    save_combined_csv(all_metric_rows, csv_path)

    print("\nPreprocessing evaluation completed successfully.")
    print(f"Combined metrics: {csv_path.resolve()}")


if __name__ == "__main__":
    main()
