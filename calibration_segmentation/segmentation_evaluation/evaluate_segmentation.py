import csv
import sys
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = (
    Path(__file__).resolve().parents[2]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )


from calibration_segmentation.roi_processing import (
    process_fruit_roi
)


# ============================================================
# PATHS
# ============================================================

CURRENT_FOLDER = Path(__file__).resolve().parent

TEST_IMAGE_FOLDER = (
    CURRENT_FOLDER / "test_images"
)

GROUND_TRUTH_FOLDER = (
    CURRENT_FOLDER / "ground_truth"
)

PREDICTED_FOLDER = (
    CURRENT_FOLDER / "predicted"
)

OVERLAY_FOLDER = (
    CURRENT_FOLDER / "overlays"
)

RESULT_CSV = (
    CURRENT_FOLDER
    / "segmentation_iou_results.csv"
)


SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
}


# ============================================================
# FIND TEST IMAGES
# ============================================================

def find_test_images():

    if not TEST_IMAGE_FOLDER.exists():
        raise FileNotFoundError(
            f"Test-image folder not found: "
            f"{TEST_IMAGE_FOLDER}"
        )

    image_paths = sorted(
        path
        for path in TEST_IMAGE_FOLDER.iterdir()
        if (
            path.is_file()
            and path.suffix.lower()
            in SUPPORTED_EXTENSIONS
        )
    )

    return image_paths


# ============================================================
# LOAD GROUND-TRUTH MASK
# ============================================================

def load_ground_truth_mask(image_path):

    mask_path = (
        GROUND_TRUTH_FOLDER
        / f"{image_path.stem}.png"
    )

    if not mask_path.exists():
        raise FileNotFoundError(
            f"No ground-truth mask found for "
            f"{image_path.name}"
        )

    mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE
    )

    if mask is None:
        raise ValueError(
            f"Unable to read ground-truth mask: "
            f"{mask_path}"
        )

    # Ensure strictly binary:
    # 0   = background
    # 255 = fruit
    _, mask = cv2.threshold(
        mask,
        127,
        255,
        cv2.THRESH_BINARY
    )

    return mask, mask_path


# ============================================================
# FIND FRUIT BOUNDING BOX FROM GROUND TRUTH
# ============================================================

def get_ground_truth_bounding_box(
    ground_truth_mask,
    margin=5
):

    points = cv2.findNonZero(
        ground_truth_mask
    )

    if points is None:
        raise ValueError(
            "Ground-truth mask contains no fruit pixels."
        )

    x, y, width, height = (
        cv2.boundingRect(points)
    )

    image_height, image_width = (
        ground_truth_mask.shape[:2]
    )

    x1 = max(
        0,
        x - margin
    )

    y1 = max(
        0,
        y - margin
    )

    x2 = min(
        image_width,
        x + width + margin
    )

    y2 = min(
        image_height,
        y + height + margin
    )

    return (
        x1,
        y1,
        x2,
        y2
    )


# ============================================================
# CALCULATE IoU
# ============================================================

def calculate_iou(
    predicted_mask,
    ground_truth_mask
):

    predicted_binary = (
        predicted_mask > 0
    )

    ground_truth_binary = (
        ground_truth_mask > 0
    )

    intersection = np.logical_and(
        predicted_binary,
        ground_truth_binary
    )

    union = np.logical_or(
        predicted_binary,
        ground_truth_binary
    )

    intersection_pixels = int(
        np.count_nonzero(intersection)
    )

    union_pixels = int(
        np.count_nonzero(union)
    )

    if union_pixels == 0:
        return 0.0, 0, 0

    iou = (
        intersection_pixels
        / union_pixels
    )

    return (
        float(iou),
        intersection_pixels,
        union_pixels
    )


# ============================================================
# CREATE COMPARISON OVERLAY
# ============================================================

def create_overlay(
    roi_image,
    predicted_mask,
    ground_truth_mask
):

    overlay = roi_image.copy()

    predicted = (
        predicted_mask > 0
    )

    ground_truth = (
        ground_truth_mask > 0
    )

    true_positive = np.logical_and(
        predicted,
        ground_truth
    )

    false_positive = np.logical_and(
        predicted,
        np.logical_not(ground_truth)
    )

    false_negative = np.logical_and(
        ground_truth,
        np.logical_not(predicted)
    )

    # Green = correctly segmented
    overlay[
        true_positive
    ] = (0, 255, 0)

    # Red = segmentation included too much
    overlay[
        false_positive
    ] = (0, 0, 255)

    # Blue = segmentation missed fruit
    overlay[
        false_negative
    ] = (255, 0, 0)

    result = cv2.addWeighted(
        roi_image,
        0.45,
        overlay,
        0.55,
        0
    )

    return result


# ============================================================
# EVALUATE ONE IMAGE
# ============================================================

def evaluate_image(image_path):

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        raise ValueError(
            f"Unable to read image: "
            f"{image_path}"
        )

    (
        ground_truth_mask,
        mask_path
    ) = load_ground_truth_mask(
        image_path
    )

    # Ground truth must correspond to
    # the original test image.
    if (
        ground_truth_mask.shape[:2]
        != image.shape[:2]
    ):
        raise ValueError(
            f"Image and ground-truth dimensions "
            f"do not match for {image_path.name}.\n"
            f"Image: {image.shape[:2]}\n"
            f"Mask : {ground_truth_mask.shape[:2]}"
        )

    # --------------------------------------------------------
    # Determine ROI from ground-truth fruit location
    # --------------------------------------------------------

    bounding_box = (
        get_ground_truth_bounding_box(
            ground_truth_mask,
            margin=5
        )
    )

    x1, y1, x2, y2 = (
        bounding_box
    )

    # --------------------------------------------------------
    # Run EXISTING segmentation
    # --------------------------------------------------------

    roi_result = process_fruit_roi(
        image,
        bounding_box,
        use_watershed=False,
        global_refined_mask=None
    )

    predicted_mask = (
        roi_result["fruit_mask"]
    )

    # --------------------------------------------------------
    # Crop ground truth to exactly same ROI
    # --------------------------------------------------------

    ground_truth_roi = (
        ground_truth_mask[
            y1:y2,
            x1:x2
        ]
    )

    # Safety check
    if (
        predicted_mask.shape
        != ground_truth_roi.shape
    ):
        raise ValueError(
            "Predicted and ground-truth "
            "ROI sizes do not match.\n"
            f"Predicted: {predicted_mask.shape}\n"
            f"Ground truth: "
            f"{ground_truth_roi.shape}"
        )

    # --------------------------------------------------------
    # IoU
    # --------------------------------------------------------

    (
        iou,
        intersection_pixels,
        union_pixels
    ) = calculate_iou(
        predicted_mask,
        ground_truth_roi
    )

    # --------------------------------------------------------
    # Save evidence
    # --------------------------------------------------------

    PREDICTED_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )

    OVERLAY_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )

    predicted_path = (
        PREDICTED_FOLDER
        / f"{image_path.stem}_predicted.png"
    )

    cv2.imwrite(
        str(predicted_path),
        predicted_mask
    )

    overlay = create_overlay(
        roi_result["roi_image"],
        predicted_mask,
        ground_truth_roi
    )

    overlay_path = (
        OVERLAY_FOLDER
        / f"{image_path.stem}_overlay.png"
    )

    cv2.imwrite(
        str(overlay_path),
        overlay
    )

    return {
        "image": image_path.name,
        "ground_truth": mask_path.name,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "predicted_pixels": int(
            np.count_nonzero(
                predicted_mask
            )
        ),
        "ground_truth_pixels": int(
            np.count_nonzero(
                ground_truth_roi
            )
        ),
        "intersection_pixels": (
            intersection_pixels
        ),
        "union_pixels": (
            union_pixels
        ),
        "iou": iou,
    }


# ============================================================
# SAVE CSV
# ============================================================

def save_results_csv(results):

    fieldnames = [
        "image",
        "ground_truth",
        "x1",
        "y1",
        "x2",
        "y2",
        "predicted_pixels",
        "ground_truth_pixels",
        "intersection_pixels",
        "union_pixels",
        "iou",
    ]

    with RESULT_CSV.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            results
        )


# ============================================================
# MAIN
# ============================================================

def main():

    image_paths = (
        find_test_images()
    )

    if len(image_paths) == 0:
        print(
            "No test images found."
        )
        return

    print()
    print(
        "Segmentation IoU Evaluation"
    )
    print(
        "=============================="
    )

    results = []

    for index, image_path in enumerate(
        image_paths,
        start=1
    ):

        print()
        print(
            f"[{index}/{len(image_paths)}] "
            f"{image_path.name}"
        )

        try:

            result = evaluate_image(
                image_path
            )

            results.append(
                result
            )

            print(
                f"IoU: "
                f"{result['iou']:.4f}"
            )

        except Exception as error:

            print(
                f"FAILED: {error}"
            )

    if len(results) == 0:
        print()
        print(
            "No images were successfully evaluated."
        )
        return

    save_results_csv(
        results
    )

    mean_iou = float(
        np.mean(
            [
                result["iou"]
                for result in results
            ]
        )
    )

    print()
    print(
        "Evaluation Summary"
    )
    print(
        "=============================="
    )

    print(
        f"Successfully evaluated : "
        f"{len(results)}"
    )

    print(
        f"Mean IoU               : "
        f"{mean_iou:.4f}"
    )

    if (
        len(results) >= 30
        and mean_iou >= 0.70
    ):

        print(
            "Objective 2 segmentation "
            "target: ACHIEVED"
        )

    else:

        print(
            "Objective 2 segmentation "
            "target: NOT YET ACHIEVED"
        )

        if len(results) < 30:
            print(
                f"Need at least 30 masks. "
                f"Currently: {len(results)}"
            )

        if mean_iou < 0.70:
            print(
                "Mean IoU must be at least "
                "0.70."
            )

    print()
    print(
        f"CSV saved to:\n"
        f"{RESULT_CSV}"
    )


if __name__ == "__main__":
    main()