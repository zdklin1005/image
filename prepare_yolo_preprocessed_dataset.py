from pathlib import Path
import shutil
import cv2

from preprocessing import (
    preprocess_fruit_image
)


# ============================================================
# DATASET PATHS
# ============================================================

SOURCE_DATASET = Path(
    r"C:\Users\winni\Downloads\FruitRipenessObjectDetection_Original"
)

OUTPUT_DATASET = Path(
    r"C:\Users\winni\Downloads\FruitRipenessObjectDetection_Preprocessed"
)


# ============================================================
# CONVERT YOLO LABEL AFTER RESIZE + PADDING
# ============================================================

def convert_yolo_label(
    source_label,
    output_label,
    original_width,
    original_height,
    resize_scale,
    resize_padding,
    output_size
):

    left, top, right, bottom = resize_padding

    output_width, output_height = output_size

    new_lines = []

    with open(
        source_label,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            parts = line.strip().split()

            if len(parts) != 5:
                continue

            class_id = parts[0]

            x_center = float(parts[1])
            y_center = float(parts[2])
            box_width = float(parts[3])
            box_height = float(parts[4])

            # ================================================
            # Convert original YOLO coordinates to pixels
            # ================================================

            x_center_pixels = (
                x_center * original_width
            )

            y_center_pixels = (
                y_center * original_height
            )

            box_width_pixels = (
                box_width * original_width
            )

            box_height_pixels = (
                box_height * original_height
            )

            # ================================================
            # Apply resizing
            # ================================================

            x_center_resized = (
                x_center_pixels
                * resize_scale
            )

            y_center_resized = (
                y_center_pixels
                * resize_scale
            )

            box_width_resized = (
                box_width_pixels
                * resize_scale
            )

            box_height_resized = (
                box_height_pixels
                * resize_scale
            )

            # ================================================
            # Apply letterbox padding
            # ================================================

            x_center_padded = (
                x_center_resized
                + left
            )

            y_center_padded = (
                y_center_resized
                + top
            )

            # ================================================
            # Convert back to normalized YOLO coordinates
            # ================================================

            new_x_center = (
                x_center_padded
                / output_width
            )

            new_y_center = (
                y_center_padded
                / output_height
            )

            new_box_width = (
                box_width_resized
                / output_width
            )

            new_box_height = (
                box_height_resized
                / output_height
            )

            new_lines.append(
                f"{class_id} "
                f"{new_x_center:.6f} "
                f"{new_y_center:.6f} "
                f"{new_box_width:.6f} "
                f"{new_box_height:.6f}\n"
            )

    with open(
        output_label,
        "w",
        encoding="utf-8"
    ) as file:

        file.writelines(new_lines)


# ============================================================
# PROCESS ONE SPLIT
# ============================================================

def process_split(split_name):

    source_images = (
        SOURCE_DATASET
        / split_name
        / "images"
    )

    source_labels = (
        SOURCE_DATASET
        / split_name
        / "labels"
    )

    output_images = (
        OUTPUT_DATASET
        / split_name
        / "images"
    )

    output_labels = (
        OUTPUT_DATASET
        / split_name
        / "labels"
    )

    output_images.mkdir(
        parents=True,
        exist_ok=True
    )

    output_labels.mkdir(
        parents=True,
        exist_ok=True
    )

    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp"
    }

    image_files = [
        file
        for file in source_images.iterdir()
        if file.suffix.lower()
        in image_extensions
    ]

    print(
        f"\nProcessing {split_name}: "
        f"{len(image_files)} images"
    )

    for index, image_path in enumerate(
        image_files,
        start=1
    ):

        try:

            # ================================================
            # RUN MEMBER 1 PREPROCESSING
            # ================================================

            preprocessing_results = (
                preprocess_fruit_image(
                    str(image_path)
                )
            )

            # IMPORTANT:
            # main.py uses classification_image for YOLO
            classification_image = (
                preprocessing_results[
                    "classification_image"
                ]
            )

            resize_scale = (
                preprocessing_results[
                    "resize_scale"
                ]
            )

            resize_padding = (
                preprocessing_results[
                    "resize_padding"
                ]
            )

            output_size = (
                preprocessing_results[
                    "output_size"
                ]
            )

            original_image = (
                preprocessing_results[
                    "source_image_full_resolution"
                ]
            )

            original_height, original_width = (
                original_image.shape[:2]
            )

            # ================================================
            # SAVE PREPROCESSED IMAGE
            # ================================================

            output_image_path = (
                output_images
                / image_path.name
            )

            success = cv2.imwrite(
                str(output_image_path),
                classification_image
            )

            if not success:

                print(
                    f"Failed to save: "
                    f"{image_path.name}"
                )

                continue

            # ================================================
            # CONVERT YOLO LABEL
            # ================================================

            label_name = (
                image_path.stem
                + ".txt"
            )

            source_label = (
                source_labels
                / label_name
            )

            output_label = (
                output_labels
                / label_name
            )

            if source_label.exists():

                convert_yolo_label(
                    source_label=source_label,
                    output_label=output_label,
                    original_width=original_width,
                    original_height=original_height,
                    resize_scale=resize_scale,
                    resize_padding=resize_padding,
                    output_size=output_size
                )

            else:

                print(
                    f"Warning: no label for "
                    f"{image_path.name}"
                )

            print(
                f"{index}/{len(image_files)} "
                f"{image_path.name}"
            )

        except Exception as error:

            print(
                f"Error processing "
                f"{image_path.name}: "
                f"{error}"
            )


# ============================================================
# COPY data.yaml
# ============================================================

def copy_data_yaml():

    source_yaml = (
        SOURCE_DATASET
        / "data.yaml"
    )

    output_yaml = (
        OUTPUT_DATASET
        / "data.yaml"
    )

    if source_yaml.exists():

        shutil.copy2(
            source_yaml,
            output_yaml
        )

        print(
            "\nCopied: data.yaml"
        )

    else:

        print(
            "\nWarning: "
            "data.yaml not found."
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    process_split("train")
    process_split("valid")
    process_split("test")

    copy_data_yaml()

    print(
        "\nPreprocessed dataset completed."
    )
    