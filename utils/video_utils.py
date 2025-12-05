"""Video utility functions."""
import cv2
from pathlib import Path
from typing import Optional


def get_video_info(video_path: str) -> dict:
    """
    Get video metadata.

    Args:
        video_path: Path to video file

    Returns:
        Dictionary with video metadata
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "duration": cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 0
    }

    cap.release()
    return info


def validate_video_path(video_path: str) -> bool:
    """
    Validate video path exists and is readable.

    Args:
        video_path: Path to video file

    Returns:
        True if valid, False otherwise
    """
    path = Path(video_path)
    if not path.exists():
        return False

    cap = cv2.VideoCapture(video_path)
    is_valid = cap.isOpened()
    cap.release()

    return is_valid


def frame_to_timestamp(frame_idx: int, fps: float) -> float:
    """
    Convert frame index to timestamp in seconds.

    Args:
        frame_idx: Frame index
        fps: Frames per second

    Returns:
        Timestamp in seconds
    """
    return frame_idx / fps if fps > 0 else 0


def timestamp_to_frame(timestamp: float, fps: float) -> int:
    """
    Convert timestamp to frame index.

    Args:
        timestamp: Timestamp in seconds
        fps: Frames per second

    Returns:
        Frame index
    """
    return int(timestamp * fps)
