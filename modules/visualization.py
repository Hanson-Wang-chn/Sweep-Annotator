"""Visualization utilities for rendering."""
import cv2
import numpy as np
from typing import List, Tuple
from config.primitives_config import TrajectorySegment


def draw_segment_timeline(image: np.ndarray, segments: List[TrajectorySegment],
                         current_frame: int, total_frames: int) -> np.ndarray:
    """
    Draw timeline visualization showing segments.

    Args:
        image: Input image to draw on
        segments: List of trajectory segments
        current_frame: Current frame index
        total_frames: Total number of frames

    Returns:
        Image with timeline overlay
    """
    height, width = image.shape[:2]
    overlay = image.copy()

    # Timeline bar dimensions
    bar_height = 20
    bar_margin = 10
    bar_y = height - bar_height - bar_margin
    bar_x_start = bar_margin
    bar_x_end = width - bar_margin
    bar_width = bar_x_end - bar_x_start

    # Draw timeline background
    cv2.rectangle(overlay, (bar_x_start, bar_y),
                 (bar_x_end, bar_y + bar_height), (50, 50, 50), -1)

    # Draw segments
    colors = [
        (255, 100, 100),  # Red
        (100, 255, 100),  # Green
        (100, 100, 255),  # Blue
        (255, 255, 100),  # Yellow
        (255, 100, 255),  # Magenta
    ]

    for i, segment in enumerate(segments):
        color = colors[i % len(colors)]
        segment_start_x = int(bar_x_start + (segment.start_frame / total_frames) * bar_width)
        segment_end_x = int(bar_x_start + (segment.end_frame / total_frames) * bar_width)

        cv2.rectangle(overlay, (segment_start_x, bar_y),
                     (segment_end_x, bar_y + bar_height), color, -1)

    # Draw current frame indicator
    current_x = int(bar_x_start + (current_frame / total_frames) * bar_width)
    cv2.line(overlay, (current_x, bar_y - 5),
            (current_x, bar_y + bar_height + 5), (255, 255, 255), 2)

    return overlay


def add_text_overlay(image: np.ndarray, text: str, position: Tuple[int, int] = (10, 30),
                    font_scale: float = 0.7, color: Tuple[int, int, int] = (255, 255, 255),
                    thickness: int = 2) -> np.ndarray:
    """
    Add text overlay to image.

    Args:
        image: Input image
        text: Text to display
        position: (x, y) position for text
        font_scale: Font scale
        color: Text color (RGB)
        thickness: Text thickness

    Returns:
        Image with text overlay
    """
    overlay = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Draw text with black outline for visibility
    cv2.putText(overlay, text, position, font, font_scale, (0, 0, 0), thickness + 2)
    cv2.putText(overlay, text, position, font, font_scale, color, thickness)

    return overlay


def create_side_by_side_view(original: np.ndarray, corrected: np.ndarray,
                             labels: Tuple[str, str] = ("Original", "Corrected")) -> np.ndarray:
    """
    Create side-by-side comparison view.

    Args:
        original: Original image
        corrected: Corrected image
        labels: Labels for each view

    Returns:
        Combined side-by-side image
    """
    # Resize images to same height if needed
    h1, w1 = original.shape[:2]
    h2, w2 = corrected.shape[:2]

    target_height = min(h1, h2)
    if h1 != target_height:
        original = cv2.resize(original, (int(w1 * target_height / h1), target_height))
    if h2 != target_height:
        corrected = cv2.resize(corrected, (int(w2 * target_height / h2), target_height))

    # Concatenate horizontally
    combined = np.hstack([original, corrected])

    # Add labels
    combined = add_text_overlay(combined, labels[0], (10, 30))
    combined = add_text_overlay(combined, labels[1], (original.shape[1] + 10, 30))

    return combined


def resize_for_display(image: np.ndarray, max_width: int = 800, max_height: int = 600) -> np.ndarray:
    """
    Resize image for display while maintaining aspect ratio.

    Args:
        image: Input image
        max_width: Maximum width
        max_height: Maximum height

    Returns:
        Resized image
    """
    h, w = image.shape[:2]

    # Calculate scaling factor
    scale = min(max_width / w, max_height / h, 1.0)

    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return image
