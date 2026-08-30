import cv2
import numpy as np
from pathlib import Path

"""
RUN THIS FILE COMMAND TO CREATE GROUND_TRUTH MASKS FOR SGEMENTATION EVALUATION:
python calibration_segmentation/segmentation_evaluation/create_ground_truth.py

"""
points = []


def mouse_callback(event, x, y, flags, param):
    global points

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))


def create_ground_truth_mask(image_path, output_path):
    global points

    points = []

    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(
            f"Unable to read image: {image_path}"
        )

    display = image.copy()

    window_name = "Ground Truth Annotation"

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(
        window_name,
        mouse_callback
    )

    print()
    print("Ground Truth Annotation")
    print("------------------------------")
    print("Left click : Add boundary point")
    print("R          : Reset points")
    print("S          : Save mask")
    print("ESC        : Cancel")

    while True:

        display = image.copy()

        if len(points) > 0:

            for point in points:
                cv2.circle(
                    display,
                    point,
                    3,
                    (0, 0, 255),
                    -1
                )

        if len(points) > 1:

            cv2.polylines(
                display,
                [
                    np.array(
                        points,
                        dtype=np.int32
                    )
                ],
                False,
                (0, 255, 0),
                2
            )

        cv2.imshow(
            window_name,
            display
        )

        key = cv2.waitKey(20) & 0xFF

        if key == ord("r"):

            points = []

        elif key == ord("s"):

            if len(points) < 3:

                print(
                    "At least 3 points are required."
                )

                continue

            mask = np.zeros(
                image.shape[:2],
                dtype=np.uint8
            )

            polygon = np.array(
                points,
                dtype=np.int32
            )

            cv2.fillPoly(
                mask,
                [polygon],
                255
            )

            output_path.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            cv2.imwrite(
                str(output_path),
                mask
            )

            print(
                f"Ground-truth mask saved: "
                f"{output_path}"
            )

            break

        elif key == 27:

            print("Annotation cancelled.")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":

    project_folder = (
        Path(__file__).resolve().parent
    )

    image_path = (
        project_folder
        / "test_images"
        / "pear_05.jpg"
    )

    output_path = (
        project_folder
        / "ground_truth"
        / "pear_05.png"
    )

    create_ground_truth_mask(
        image_path,
        output_path
    )