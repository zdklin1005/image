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

    # Create output folders automatically
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
        if file.suffix.lower() in image_extensions
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
            # Run teammate preprocessing
            preprocessing_results = (
                preprocess_fruit_image(
                    str(image_path)
                )
            )

            # Use analysis_image
            analysis_image = (
                preprocessing_results[
                    "analysis_image"
                ]
            )

            # Save preprocessed image
            output_image_path = (
                output_images
                / image_path.name
            )

            cv2.imwrite(
                str(output_image_path),
                analysis_image
            )

            # Copy original YOLO label
            label_name = (
                image_path.stem + ".txt"
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
                shutil.copy2(
                    source_label,
                    output_label
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

    source_yaml = SOURCE_DATASET / "data.yaml"
    output_yaml = OUTPUT_DATASET / "data.yaml"

    if source_yaml.exists():
        shutil.copy2(
            source_yaml,
            output_yaml
        )

        print("\nCopied: data.yaml")

    else:
        print(
            f"\nWarning: data.yaml not found at "
            f"{source_yaml}"
        )

# ============================================================
# PROCESS TRAIN / VALID / TEST
# ============================================================
if __name__ == "__main__":
    process_split("train")
    process_split("valid")
    process_split("test")

    # Copy data.yaml
    copy_data_yaml()

    print(
        "\nPreprocessed dataset completed."
    )
