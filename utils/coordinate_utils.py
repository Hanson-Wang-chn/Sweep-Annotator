"""Coordinate transformation utilities."""
import numpy as np
from typing import Tuple
from config.primitives_config import Coordinate


def pixel_to_normalized(x: int, y: int, image_size: Tuple[int, int]) -> Coordinate:
    """
    Convert pixel coordinates to normalized [0, 1] coordinates.

    Args:
        x, y: Pixel coordinates
        image_size: (width, height) of image

    Returns:
        Normalized coordinate (floats in range [0, 1] with 3 decimal places)
    """
    width, height = image_size
    norm_x = round(x / width, 3)
    norm_y = round(y / height, 3)
    return Coordinate(x=norm_x, y=norm_y)


def normalized_to_pixel(coord: Coordinate, image_size: Tuple[int, int]) -> Tuple[int, int]:
    """
    Convert normalized [0, 1] coordinates to pixel coordinates.

    Args:
        coord: Normalized coordinate (floats in range [0, 1])
        image_size: (width, height) of image

    Returns:
        (x, y) pixel coordinates
    """
    width, height = image_size
    pixel_x = int(round(coord.x * width))
    pixel_y = int(round(coord.y * height))
    return (pixel_x, pixel_y)


def clip_coordinate(coord: Coordinate, min_val: float = 0.0, max_val: float = 1.0) -> Coordinate:
    """
    Clip coordinate values to range.

    Args:
        coord: Input coordinate
        min_val: Minimum value (default: 0.0)
        max_val: Maximum value (default: 1.0)

    Returns:
        Clipped coordinate
    """
    return Coordinate(
        x=float(np.clip(coord.x, min_val, max_val)),
        y=float(np.clip(coord.y, min_val, max_val))
    )


def compute_bbox(coords: list) -> Tuple[Coordinate, Coordinate]:
    """
    Compute bounding box from coordinates.

    Args:
        coords: List of Coordinate objects

    Returns:
        (top_left, bottom_right) coordinates
    """
    if not coords:
        return Coordinate(0.0, 0.0), Coordinate(1.0, 1.0)

    xs = [c.x for c in coords]
    ys = [c.y for c in coords]

    return Coordinate(min(xs), min(ys)), Coordinate(max(xs), max(ys))
