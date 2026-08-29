"""Prepare overlapping image tiles for high-resolution YOLO inference."""

import cv2

from .preprocessing import (
    add_letterbox_padding,
    create_valid_content_mask,
    resize_preserving_aspect_ratio,
)


def _axis_starts(length, tile_length, overlap_ratio):
    if length <= tile_length:
        return [0]

    stride = max(1, int(round(tile_length * (1.0 - overlap_ratio))))
    starts = list(range(0, length - tile_length + 1, stride))
    final_start = length - tile_length

    if starts[-1] != final_start:
        starts.append(final_start)

    return starts


def _ownership_interval(starts, index, tile_length, image_length):
    start = starts[index]
    end = min(start + tile_length, image_length)

    if index == 0:
        ownership_start = 0.0
    else:
        previous_end = min(starts[index - 1] + tile_length, image_length)
        ownership_start = (start + previous_end) / 2.0

    if index == len(starts) - 1:
        ownership_end = float(image_length)
    else:
        next_start = starts[index + 1]
        ownership_end = (end + next_start) / 2.0

    return ownership_start, ownership_end


def create_overlapping_tiles(
    image,
    tile_size=(640, 640),
    overlap_ratio=0.20,
    output_size=(640, 640),
    median_kernel=5,
    bilateral_diameter=5,
    bilateral_sigma_colour=25,
    bilateral_sigma_space=25,
    skip_if_image_fits=True,
):
    # Divide a high-resolution image into overlapping, preprocessed tiles.
    if image is None or image.size == 0:
        raise ValueError("image must contain pixel data.")

    if not 0.0 <= overlap_ratio < 1.0:
        raise ValueError("overlap_ratio must be at least 0 and below 1.")

    if median_kernel <= 0 or median_kernel % 2 == 0:
        raise ValueError("median_kernel must be a positive odd integer.")

    tile_width, tile_height = tile_size
    image_height, image_width = image.shape[:2]

    if tile_width <= 0 or tile_height <= 0:
        raise ValueError("tile_size values must be positive.")

    if (
        skip_if_image_fits
        and image_width <= tile_width
        and image_height <= tile_height
    ):
        return []

    x_starts = _axis_starts(image_width, tile_width, overlap_ratio)
    y_starts = _axis_starts(image_height, tile_height, overlap_ratio)
    tiles = []

    for y_index, y1 in enumerate(y_starts):
        for x_index, x1 in enumerate(x_starts):
            x2 = min(x1 + tile_width, image_width)
            y2 = min(y1 + tile_height, image_height)
            source_tile = image[y1:y2, x1:x2].copy()

            resized_tile, tile_scale, tile_padding = (
                resize_preserving_aspect_ratio(source_tile, output_size)
            )
            median_tile = cv2.medianBlur(resized_tile, median_kernel)
            classification_tile = cv2.bilateralFilter(
                median_tile,
                bilateral_diameter,
                bilateral_sigma_colour,
                bilateral_sigma_space,
            )
            classification_tile = add_letterbox_padding(
                classification_tile,
                output_size,
            )

            ownership_x1, ownership_x2 = _ownership_interval(
                x_starts,
                x_index,
                tile_width,
                image_width,
            )
            ownership_y1, ownership_y2 = _ownership_interval(
                y_starts,
                y_index,
                tile_height,
                image_height,
            )

            tiles.append({
                "classification_image": classification_tile,
                "valid_content_mask": create_valid_content_mask(
                    output_size,
                    tile_padding,
                ),
                "source_box": (x1, y1, x2, y2),
                "ownership_box": (
                    ownership_x1,
                    ownership_y1,
                    ownership_x2,
                    ownership_y2,
                ),
                "resize_scale": tile_scale,
                "resize_padding": tile_padding,
                "output_size": output_size,
            })

    return tiles


def map_tile_detection_to_standard_image(
    detection,
    tile,
    full_resize_scale,
    full_resize_padding,
    standardised_size=(640, 640),
):
    # Map one tile-local YOLO box into the full standardised image.
    x1, y1, x2, y2 = detection["bounding_box"]
    left_padding, top_padding, _, _ = tile["resize_padding"]
    tile_scale = tile["resize_scale"]
    source_x1, source_y1, source_x2, source_y2 = tile["source_box"]

    local_x1 = (x1 - left_padding) / tile_scale
    local_y1 = (y1 - top_padding) / tile_scale
    local_x2 = (x2 - left_padding) / tile_scale
    local_y2 = (y2 - top_padding) / tile_scale

    local_x1 = min(max(local_x1, 0.0), source_x2 - source_x1)
    local_y1 = min(max(local_y1, 0.0), source_y2 - source_y1)
    local_x2 = min(max(local_x2, 0.0), source_x2 - source_x1)
    local_y2 = min(max(local_y2, 0.0), source_y2 - source_y1)

    full_x1 = source_x1 + local_x1
    full_y1 = source_y1 + local_y1
    full_x2 = source_x1 + local_x2
    full_y2 = source_y1 + local_y2
    centre_x = (full_x1 + full_x2) / 2.0
    centre_y = (full_y1 + full_y2) / 2.0
    owner_x1, owner_y1, owner_x2, owner_y2 = tile["ownership_box"]

    if not (
        owner_x1 <= centre_x < owner_x2
        and owner_y1 <= centre_y < owner_y2
    ):
        return None

    full_left_padding, full_top_padding, _, _ = full_resize_padding
    standard_width, standard_height = standardised_size
    mapped_box = (
        max(0, int(round(full_x1 * full_resize_scale + full_left_padding))),
        max(0, int(round(full_y1 * full_resize_scale + full_top_padding))),
        min(
            standard_width,
            int(round(full_x2 * full_resize_scale + full_left_padding)),
        ),
        min(
            standard_height,
            int(round(full_y2 * full_resize_scale + full_top_padding)),
        ),
    )

    if mapped_box[2] <= mapped_box[0] or mapped_box[3] <= mapped_box[1]:
        return None

    mapped_detection = dict(detection)
    mapped_detection["bounding_box"] = mapped_box
    mapped_detection["detection_pass"] = "tiled"
    return mapped_detection


def map_tile_detections_to_standard_image(
    detections,
    tile,
    full_resize_scale,
    full_resize_padding,
    standardised_size=(640, 640),
):
    mapped = []

    for detection in detections:
        mapped_detection = map_tile_detection_to_standard_image(
            detection,
            tile,
            full_resize_scale,
            full_resize_padding,
            standardised_size,
        )

        if mapped_detection is not None:
            mapped.append(mapped_detection)

    return mapped
