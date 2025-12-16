"""Coordinate transformation utilities."""
import numpy as np
from typing import Tuple
from config.primitives_config import Coordinate


def pixel_to_normalized(x: int, y: int, image_size: Tuple[int, int]) -> Coordinate:
    """
    Convert pixel coordinates to normalized [0, 1000] coordinates.

    Args:
        x, y: Pixel coordinates
        image_size: (width, height) of image

    Returns:
        Normalized coordinate (integers in range [0, 1000])
    """
    width, height = image_size
    norm_x = int(round((x / width) * 1000))
    norm_y = int(round((y / height) * 1000))
    return Coordinate(x=norm_x, y=norm_y)


def normalized_to_pixel(coord: Coordinate, image_size: Tuple[int, int]) -> Tuple[int, int]:
    """
    Convert normalized [0, 1000] coordinates to pixel coordinates.

    Args:
        coord: Normalized coordinate (integers in range [0, 1000])
        image_size: (width, height) of image

    Returns:
        (x, y) pixel coordinates
    """
    width, height = image_size
    pixel_x = int(round((coord.x / 1000) * width))
    pixel_y = int(round((coord.y / 1000) * height))
    return (pixel_x, pixel_y)


def clip_coordinate(coord: Coordinate, min_val: int = 0, max_val: int = 1000) -> Coordinate:
    """
    Clip coordinate values to range.

    Args:
        coord: Input coordinate
        min_val: Minimum value (default: 0)
        max_val: Maximum value (default: 1000)

    Returns:
        Clipped coordinate
    """
    return Coordinate(
        x=int(np.clip(coord.x, min_val, max_val)),
        y=int(np.clip(coord.y, min_val, max_val))
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
        return Coordinate(0, 0), Coordinate(1000, 1000)

    xs = [c.x for c in coords]
    ys = [c.y for c in coords]

    return Coordinate(min(xs), min(ys)), Coordinate(max(xs), max(ys))
