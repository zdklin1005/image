from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PRIMARY_ROOT = PROJECT_ROOT.parent / "dataset" / "Test"
DEFAULT_ARCHIVE_ROOT = PROJECT_ROOT.parent / "archive (1)"
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "preprocessing_evaluation" / "clahe_comparison"
)

PRIMARY_FOLDERS = (
    "freshapples",
    "freshbanana",
    "freshoranges",
    "rottenapples",
    "rottenbanana",
    "rottenoranges",
)

ARCHIVE_FOLDERS = (
    "Ripe Apple",
    "Unripe Apple",
    "Ripe Banana",
    "Unripe Banana",
    "Ripe Grape",
    "Unripe Grape",
    "Ripe Mango",
    "Unripe Mango",
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def evenly_spaced_files(folder: Path, count: int) -> list[Path]:
    files = sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )

    if not files:
        return []

    if len(files) <= count:
        return files

    positions = np.linspace(0, len(files) - 1, count)
    indices = sorted({int(round(position)) for position in positions})
    return [files[index] for index in indices]


def select_images(
    primary_root: Path,
    archive_root: Path,
    primary_per_folder: int,
    archive_per_folder: int,
) -> list[tuple[str, str, Path]]:
    selected: list[tuple[str, str, Path]] = []

    for folder_name in PRIMARY_FOLDERS:
        folder = primary_root / folder_name
        if folder.is_dir():
            selected.extend(
                ("primary", folder_name, path)
                for path in evenly_spaced_files(folder, primary_per_folder)
            )

    for folder_name in ARCHIVE_FOLDERS:
        folder = archive_root / folder_name
        if folder.is_dir():
            selected.extend(
                ("archive", folder_name, path)
                for path in evenly_spaced_files(folder, archive_per_folder)
            )

    return selected


def resize_content(
    image: np.ndarray,
    output_size: tuple[int, int] = (640, 640),
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    output_width, output_height = output_size
    image_height, image_width = image.shape[:2]
    scale = min(output_width / image_width, output_height / image_height)

    resized_width = max(1, int(round(image_width * scale)))
    resized_height = max(1, int(round(image_height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR

    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=interpolation,
    )

    left = (output_width - resized_width) // 2
    top = (output_height - resized_height) // 2
    right = output_width - resized_width - left
    bottom = output_height - resized_height - top
    return resized, (left, top, right, bottom)


def add_padding(
    image: np.ndarray,
    padding: tuple[int, int, int, int],
    output_size: tuple[int, int] = (640, 640),
) -> np.ndarray:
    output_width, output_height = output_size
    left, top, right, bottom = padding
    canvas = np.full(
        (output_height, output_width, 3),
        255,
        dtype=np.uint8,
    )
    canvas[top:output_height - bottom, left:output_width - right] = image
    return canvas


def make_valid_content_mask(
    padding: tuple[int, int, int, int],
    output_size: tuple[int, int] = (640, 640),
) -> np.ndarray:
    output_width, output_height = output_size
    left, top, right, bottom = padding
    mask = np.zeros((output_height, output_width), dtype=np.uint8)
    mask[top:output_height - bottom, left:output_width - right] = 255
    return mask


def enhance_hsv_value(
    bilateral: np.ndarray,
    clip_limit: float = 1.0,
    grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    hsv = cv2.cvtColor(bilateral, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    enhanced_value = clahe.apply(value)
    return cv2.cvtColor(
        cv2.merge((hue, saturation, enhanced_value)),
        cv2.COLOR_HSV2BGR,
    )


def enhance_lab_lightness(
    bilateral: np.ndarray,
    clip_limit: float = 1.0,
    grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    lab = cv2.cvtColor(bilateral, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    enhanced_lightness = clahe.apply(lightness)
    return cv2.cvtColor(
        cv2.merge((enhanced_lightness, channel_a, channel_b)),
        cv2.COLOR_LAB2BGR,
    )


def refine_mask(mask: np.ndarray) -> np.ndarray:
    opening_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closing_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, opening_kernel)
    return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, closing_kernel)


def segment_like_current_pipeline(
    image: np.ndarray,
    valid_content_mask: np.ndarray,
) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, gray_mask = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    saturation = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 1]
    _, saturation_mask = cv2.threshold(
        saturation,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    gray_refined = refine_mask(gray_mask)
    saturation_refined = refine_mask(saturation_mask)

    expansion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    expanded_saturation = cv2.dilate(
        saturation_refined,
        expansion_kernel,
        iterations=1,
    )
    recovered_gray = cv2.bitwise_and(gray_refined, expanded_saturation)
    combined = cv2.bitwise_or(saturation_refined, recovered_gray)
    refined = refine_mask(combined)
    refined = cv2.bitwise_and(refined, valid_content_mask)

    contours, _ = cv2.findContours(
        refined,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    final_mask = np.zeros_like(refined)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 0:
            cv2.drawContours(final_mask, [largest], -1, 255, cv2.FILLED)

    return final_mask


def mask_metrics(
    mask: np.ndarray,
    valid_content_mask: np.ndarray,
) -> dict[str, float]:
    valid_pixels = max(1, cv2.countNonZero(valid_content_mask))
    foreground_pixels = cv2.countNonZero(mask)
    area_ratio = foreground_pixels / valid_pixels

    ys, xs = np.where(valid_content_mask > 0)
    if len(xs) == 0 or foreground_pixels == 0:
        return {
            "area_ratio": area_ratio,
            "border_contact_ratio": 1.0,
            "centre_coverage": 0.0,
            "solidity": 0.0,
            "proxy_score": 0.0,
            "proxy_valid": 0.0,
        }

    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    border = np.zeros_like(mask)
    border[y1, x1:x2 + 1] = 255
    border[y2, x1:x2 + 1] = 255
    border[y1:y2 + 1, x1] = 255
    border[y1:y2 + 1, x2] = 255
    border_pixels = max(1, cv2.countNonZero(border))
    border_contact = cv2.countNonZero(cv2.bitwise_and(mask, border))
    border_contact_ratio = border_contact / border_pixels

    content_width = x2 - x1 + 1
    content_height = y2 - y1 + 1
    cx1 = x1 + content_width // 4
    cx2 = x2 - content_width // 4
    cy1 = y1 + content_height // 4
    cy2 = y2 - content_height // 4
    centre = mask[cy1:cy2 + 1, cx1:cx2 + 1]
    centre_coverage = cv2.countNonZero(centre) / max(1, centre.size)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    solidity = 0.0
    if contours:
        contour = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            solidity = cv2.contourArea(contour) / hull_area

    if area_ratio < 0.02 or area_ratio > 0.90:
        area_score = 0.0
    elif area_ratio <= 0.35:
        area_score = min(1.0, (area_ratio - 0.02) / 0.20)
    else:
        area_score = max(0.0, 1.0 - (area_ratio - 0.35) / 0.55)

    proxy_score = (
        0.35 * area_score
        + 0.25 * (1.0 - border_contact_ratio)
        + 0.20 * centre_coverage
        + 0.20 * solidity
    )
    proxy_valid = float(
        0.02 <= area_ratio <= 0.90 and border_contact_ratio < 0.80
    )

    return {
        "area_ratio": area_ratio,
        "border_contact_ratio": border_contact_ratio,
        "centre_coverage": centre_coverage,
        "solidity": solidity,
        "proxy_score": proxy_score,
        "proxy_valid": proxy_valid,
    }


def colour_and_contrast_metrics(
    baseline: np.ndarray,
    enhanced: np.ndarray,
) -> dict[str, float]:
    baseline_hsv = cv2.cvtColor(baseline, cv2.COLOR_BGR2HSV)
    enhanced_hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
    saturated = baseline_hsv[:, :, 1] >= 25

    hue_difference = np.abs(
        baseline_hsv[:, :, 0].astype(np.int16)
        - enhanced_hsv[:, :, 0].astype(np.int16)
    )
    hue_difference = np.minimum(hue_difference, 180 - hue_difference)

    if np.any(saturated):
        mean_hue_drift = float(np.mean(hue_difference[saturated]))
    else:
        mean_hue_drift = 0.0

    saturation_drift = float(
        np.mean(
            np.abs(
                baseline_hsv[:, :, 1].astype(np.int16)
                - enhanced_hsv[:, :, 1].astype(np.int16)
            )
        )
    )

    baseline_lightness = cv2.cvtColor(baseline, cv2.COLOR_BGR2LAB)[:, :, 0]
    enhanced_lightness = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)[:, :, 0]

    contrast_before = float(np.std(baseline_lightness))
    contrast_after = float(np.std(enhanced_lightness))
    contrast_gain = contrast_after - contrast_before

    clipped_ratio = float(
        np.mean((enhanced_lightness <= 2) | (enhanced_lightness >= 253))
    )

    return {
        "mean_hue_drift": mean_hue_drift,
        "mean_saturation_drift": saturation_drift,
        "lightness_contrast": contrast_after,
        "contrast_gain": contrast_gain,
        "clipped_lightness_ratio": clipped_ratio,
    }


def mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    first_binary = first > 0
    second_binary = second > 0
    union = np.logical_or(first_binary, second_binary).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(first_binary, second_binary).sum() / union)


def label_panel(image: np.ndarray, label: str) -> np.ndarray:
    output = image.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(
        output,
        label,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def create_comparison_panel(record: dict, output_path: Path) -> None:
    size = (360, 260)

    def prepared(image: np.ndarray, label: str) -> np.ndarray:
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        resized = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        return label_panel(resized, label)

    top = cv2.hconcat(
        [
            prepared(record["bilateral"], "Bilateral baseline"),
            prepared(record["hsv_image"], "HSV V-channel CLAHE"),
            prepared(record["lab_image"], "LAB L-channel CLAHE"),
        ]
    )
    bottom = cv2.hconcat(
        [
            prepared(record["valid_mask"], "Valid content"),
            prepared(record["hsv_mask"], "HSV segmentation mask"),
            prepared(record["lab_mask"], "LAB segmentation mask"),
        ]
    )
    panel = cv2.vconcat([top, bottom])
    cv2.imwrite(str(output_path), panel)


def mean(rows: list[dict], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return sum(values) / max(1, len(values))


def write_summary(rows: list[dict], output_path: Path) -> None:
    lines = [
        "HSV VALUE-CHANNEL CLAHE VS LAB LIGHTNESS-CHANNEL CLAHE",
        "=" * 60,
        "",
        f"Images evaluated: {len(rows)}",
        "Ground-truth masks: unavailable",
        "Segmentation results below are plausibility proxies, not Dice/IoU accuracy.",
        "",
    ]

    for dataset_name in ("all", "primary", "archive"):
        selected = (
            rows
            if dataset_name == "all"
            else [row for row in rows if row["dataset"] == dataset_name]
        )
        if not selected:
            continue

        lines.extend(
            [
                dataset_name.upper(),
                "-" * len(dataset_name),
                f"Images: {len(selected)}",
                f"Mean HSV/LAB mask agreement (IoU): {mean(selected, 'mask_iou'):.4f}",
                f"HSV proxy-valid masks: {100 * mean(selected, 'hsv_proxy_valid'):.2f}%",
                f"LAB proxy-valid masks: {100 * mean(selected, 'lab_proxy_valid'):.2f}%",
                f"HSV mean mask proxy score: {mean(selected, 'hsv_proxy_score'):.4f}",
                f"LAB mean mask proxy score: {mean(selected, 'lab_proxy_score'):.4f}",
                f"HSV mean border contact: {mean(selected, 'hsv_border_contact_ratio'):.4f}",
                f"LAB mean border contact: {mean(selected, 'lab_border_contact_ratio'):.4f}",
                f"HSV mean hue drift (OpenCV hue units): {mean(selected, 'hsv_mean_hue_drift'):.4f}",
                f"LAB mean hue drift (OpenCV hue units): {mean(selected, 'lab_mean_hue_drift'):.4f}",
                f"HSV mean saturation drift: {mean(selected, 'hsv_mean_saturation_drift'):.4f}",
                f"LAB mean saturation drift: {mean(selected, 'lab_mean_saturation_drift'):.4f}",
                f"HSV mean lightness contrast gain: {mean(selected, 'hsv_contrast_gain'):.4f}",
                f"LAB mean lightness contrast gain: {mean(selected, 'lab_contrast_gain'):.4f}",
                f"HSV mean processing time: {mean(selected, 'hsv_processing_ms'):.3f} ms",
                f"LAB mean processing time: {mean(selected, 'lab_processing_ms'):.3f} ms",
                "",
            ]
        )

    hsv_wins = sum(
        float(row["hsv_proxy_score"]) > float(row["lab_proxy_score"])
        for row in rows
    )
    lab_wins = sum(
        float(row["lab_proxy_score"]) > float(row["hsv_proxy_score"])
        for row in rows
    )
    ties = len(rows) - hsv_wins - lab_wins
    lines.extend(
        [
            "PROXY COMPARISON COUNTS",
            "-----------------------",
            f"HSV higher proxy score: {hsv_wins}",
            f"LAB higher proxy score: {lab_wins}",
            f"Ties: {ties}",
            "",
            "Interpret all proxy results together with the saved visual panels.",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def evaluate(args: argparse.Namespace) -> None:
    selected = select_images(
        Path(args.primary_root),
        Path(args.archive_root),
        args.primary_per_folder,
        args.archive_per_folder,
    )
    if not selected:
        raise RuntimeError("No evaluation images were found.")

    output_root = Path(args.output_root)
    panel_root = output_root / "panels"
    panel_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    panel_records: list[dict] = []

    for index, (dataset_name, folder_name, image_path) in enumerate(selected, 1):
        source = cv2.imread(str(image_path))
        if source is None:
            continue

        resized, padding = resize_content(source)
        median = cv2.medianBlur(resized, 5)
        bilateral_content = cv2.bilateralFilter(median, 5, 25, 25)

        start = time.perf_counter()
        hsv_content = enhance_hsv_value(bilateral_content)
        hsv_processing_ms = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        lab_content = enhance_lab_lightness(bilateral_content)
        lab_processing_ms = (time.perf_counter() - start) * 1000

        bilateral = add_padding(bilateral_content, padding)
        hsv_image = add_padding(hsv_content, padding)
        lab_image = add_padding(lab_content, padding)
        valid_mask = make_valid_content_mask(padding)

        hsv_mask = segment_like_current_pipeline(hsv_image, valid_mask)
        lab_mask = segment_like_current_pipeline(lab_image, valid_mask)

        hsv_mask_metrics = mask_metrics(hsv_mask, valid_mask)
        lab_mask_metrics = mask_metrics(lab_mask, valid_mask)
        hsv_colour_metrics = colour_and_contrast_metrics(
            bilateral_content,
            hsv_content,
        )
        lab_colour_metrics = colour_and_contrast_metrics(
            bilateral_content,
            lab_content,
        )

        row = {
            "dataset": dataset_name,
            "folder": folder_name,
            "image_path": str(image_path),
            "mask_iou": mask_iou(hsv_mask, lab_mask),
            "hsv_processing_ms": hsv_processing_ms,
            "lab_processing_ms": lab_processing_ms,
        }
        row.update({f"hsv_{key}": value for key, value in hsv_mask_metrics.items()})
        row.update({f"lab_{key}": value for key, value in lab_mask_metrics.items()})
        row.update({f"hsv_{key}": value for key, value in hsv_colour_metrics.items()})
        row.update({f"lab_{key}": value for key, value in lab_colour_metrics.items()})
        rows.append(row)

        panel_records.append(
            {
                "row": row,
                "bilateral": bilateral,
                "hsv_image": hsv_image,
                "lab_image": lab_image,
                "valid_mask": valid_mask,
                "hsv_mask": hsv_mask,
                "lab_mask": lab_mask,
            }
        )

        if index % 20 == 0 or index == len(selected):
            print(f"Evaluated {index}/{len(selected)} images")

    csv_path = output_root / "clahe_comparison_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_path = output_root / "clahe_comparison_summary.txt"
    write_summary(rows, summary_path)

    panel_records.sort(key=lambda record: float(record["row"]["mask_iou"]))
    for panel_index, record in enumerate(panel_records[: args.panel_count], 1):
        safe_folder = "_".join(record["row"]["folder"].split())
        panel_path = panel_root / f"{panel_index:02d}_{safe_folder}.png"
        create_comparison_panel(record, panel_path)

    print(f"Metrics: {csv_path.resolve()}")
    print(f"Summary: {summary_path.resolve()}")
    print(f"Panels: {panel_root.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare HSV Value-channel and LAB Lightness-channel CLAHE.",
    )
    parser.add_argument("--primary-root", default=str(DEFAULT_PRIMARY_ROOT))
    parser.add_argument("--archive-root", default=str(DEFAULT_ARCHIVE_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--primary-per-folder", type=int, default=15)
    parser.add_argument("--archive-per-folder", type=int, default=10)
    parser.add_argument("--panel-count", type=int, default=12)
    return parser


if __name__ == "__main__":
    evaluate(build_parser().parse_args())
