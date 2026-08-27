from pathlib import Path
import cv2

# ============================================================
# DATASET PATHS
# ============================================================
SOURCE_DATASET = Path(
    r"C:\Users\winni\Downloads\FruitRipenessObjectDetection_Original"
)

OUTPUT_DATASET = Path(
    r"C:\Users\winni\Downloads\E_RipenessClassification_4Class"
)

# ============================================================
# RIPENESS CLASS MAPPING
# ============================================================
RIPENESS_CLASSES = {
    0: "Overripe",
    1: "Ripe",
    2: "Rotten",
    3: "Unripe"
}

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

    # Create condition folders
    for class_name in RIPENESS_CLASSES.values():

        (
            OUTPUT_DATASET
            / split_name
            / class_name
        ).mkdir(
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

    for image_index, image_path in enumerate(
        image_files,
        start=1
    ):

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            print(
                f"Cannot read: {image_path.name}"
            )
            continue

        image_height, image_width = (
            image.shape[:2]
        )

        label_path = (
            source_labels
            / (image_path.stem + ".txt")
        )

        if not label_path.exists():
            continue

        with open(
            label_path,
            "r",
            encoding="utf-8"
        ) as file:

            lines = file.readlines()

        # One image may contain more than one fruit
        for object_index, line in enumerate(
            lines,
            start=1
        ):

            parts = line.strip().split()

            if len(parts) != 5:
                continue

            class_id = int(parts[0])

            x_center = float(parts[1])
            y_center = float(parts[2])
            box_width = float(parts[3])
            box_height = float(parts[4])

            # ================================================
            # GET RIPENESS CONDITION
            # ================================================
            condition_id = (
                class_id % 4
            )

            condition_name = (
                RIPENESS_CLASSES[
                    condition_id
                ]
            )

            # ================================================
            # YOLO COORDINATES -> PIXELS
            # ================================================
            center_x = (
                x_center * image_width
            )

            center_y = (
                y_center * image_height
            )

            width_pixels = (
                box_width * image_width
            )

            height_pixels = (
                box_height * image_height
            )

            x1 = int(
                center_x
                - width_pixels / 2
            )

            y1 = int(
                center_y
                - height_pixels / 2
            )

            x2 = int(
                center_x
                + width_pixels / 2
            )

            y2 = int(
                center_y
                + height_pixels / 2
            )

            # ================================================
            # ADD SMALL 10% MARGIN
            # ================================================
            margin_x = int(
                width_pixels * 0.1
            )

            margin_y = int(
                height_pixels * 0.1
            )

            x1 = max(
                0,
                x1 - margin_x
            )

            y1 = max(
                0,
                y1 - margin_y
            )

            x2 = min(
                image_width,
                x2 + margin_x
            )

            y2 = min(
                image_height,
                y2 + margin_y
            )

            # ================================================
            # CROP
            # ================================================
            fruit_crop = image[
                y1:y2,
                x1:x2
            ]

            if fruit_crop.size == 0:
                continue

            output_filename = (
                f"{image_path.stem}"
                f"_object_{object_index}.jpg"
            )

            output_path = (
                OUTPUT_DATASET
                / split_name
                / condition_name
                / output_filename
            )

            cv2.imwrite(
                str(output_path),
                fruit_crop
            )

        print(
            f"{image_index}/{len(image_files)} "
            f"{image_path.name}"
        )


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    process_split("train")
    process_split("valid")
    process_split("test")

    print(
        "\nModel E dataset completed."
    )
