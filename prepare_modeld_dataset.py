from pathlib import Path
import shutil

# ============================================================
# DATASET PATHS
# ============================================================
SOURCE_DATASET = Path(
    r"C:\Users\winni\Downloads\FruitRipenessObjectDetection_Original"
)

OUTPUT_DATASET = Path(
    r"C:\Users\winni\Downloads\D_FruitDetection_8Fruits"
)

# ============================================================
# CLASS NAMES
# ============================================================
FRUIT_NAMES = [
    "Apple",
    "Banana",
    "Grape",
    "Mango",
    "Melon",
    "Orange",
    "Peach",
    "Pear"
]

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

        # ================================================
        # COPY IMAGE
        # ================================================
        shutil.copy2(
            image_path,
            output_images / image_path.name
        )

        # ================================================
        # CONVERT LABEL
        # ================================================
        label_name = (
            image_path.stem + ".txt"
        )

        source_label = (
            source_labels / label_name
        )

        output_label = (
            output_labels / label_name
        )

        if source_label.exists():

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

                    old_class_id = int(parts[0])

                    # 32 classes -> 8 fruit classes
                    new_class_id = (
                        old_class_id // 4
                    )

                    new_line = (
                        f"{new_class_id} "
                        f"{parts[1]} "
                        f"{parts[2]} "
                        f"{parts[3]} "
                        f"{parts[4]}\n"
                    )

                    new_lines.append(
                        new_line
                    )

            with open(
                output_label,
                "w",
                encoding="utf-8"
            ) as file:

                file.writelines(
                    new_lines
                )

        print(
            f"{index}/{len(image_files)} "
            f"{image_path.name}"
        )


# ============================================================
# CREATE data.yaml
# ============================================================
def create_yaml():

    yaml_path = (
        OUTPUT_DATASET / "data.yaml"
    )

    yaml_text = """train: train/images
val: valid/images
test: test/images

nc: 8
names:
  - Apple
  - Banana
  - Grape
  - Mango
  - Melon
  - Orange
  - Peach
  - Pear
"""

    yaml_path.write_text(
        yaml_text,
        encoding="utf-8"
    )

    print("\nCreated data.yaml")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    process_split("train")
    process_split("valid")
    process_split("test")

    create_yaml()

    print(
        "\nModel D dataset completed."
    )
