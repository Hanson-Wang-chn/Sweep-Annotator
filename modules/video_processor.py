"""Video processing module for frame extraction and video segment export."""
from typing import List, Optional
import cv2
import numpy as np


class VideoProcessor:
    """Process video files for frame extraction and segment export."""

    def __init__(self, video_path: str):
        """
        Initialize video processor.

        Args:
            video_path: Path to video file
        """
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def get_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        """
        Extract single frame at index.

        Args:
            frame_idx: Frame index (0-based)

        Returns:
            RGB image as numpy array (H, W, 3) or None if failed
        """
        if frame_idx < 0 or frame_idx >= self.total_frames:
            raise ValueError(f"Frame index {frame_idx} out of range [0, {self.total_frames})")

        # Seek to frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()

        if not ret:
            return None

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame_rgb

    def get_frame_range(self, start_frame: int, end_frame: int) -> List[np.ndarray]:
        """
        Extract range of frames.

        Args:
            start_frame: Start frame index (inclusive)
            end_frame: End frame index (inclusive)

        Returns:
            List of RGB images
        """
        if start_frame < 0 or end_frame >= self.total_frames:
            raise ValueError(f"Frame range [{start_frame}, {end_frame}] out of bounds")

        if start_frame > end_frame:
            raise ValueError(f"start_frame ({start_frame}) must be <= end_frame ({end_frame})")

        frames = []
        for frame_idx in range(start_frame, end_frame + 1):
            frame = self.get_frame(frame_idx)
            if frame is not None:
                frames.append(frame)

        return frames

    def export_segment(self, start_frame: int, end_frame: int, output_path: str,
                      transform_fn=None, output_size=None):
        """
        Export video segment to new mp4 file.

        Args:
            start_frame: Start frame index
            end_frame: End frame index (inclusive)
            output_path: Output video file path
            transform_fn: Optional transformation function to apply to each frame
            output_size: Optional output size (width, height) for transformed frames
        """
        if start_frame < 0 or end_frame >= self.total_frames:
            raise ValueError(f"Frame range [{start_frame}, {end_frame}] out of bounds")

        # Determine output dimensions
        if output_size:
            out_width, out_height = output_size
        else:
            out_width, out_height = self.width, self.height

        # Create video writer with high quality settings
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (out_width, out_height))

        if not out.isOpened():
            raise RuntimeError(f"Failed to create output video: {output_path}")

        # Write frames
        for frame_idx in range(start_frame, end_frame + 1):
            frame = self.get_frame(frame_idx)
            if frame is None:
                continue

            # Apply transformation if provided
            if transform_fn:
                frame = transform_fn(frame)

            # Resize if needed
            if frame.shape[:2] != (out_height, out_width):
                frame = cv2.resize(frame, (out_width, out_height))

            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)

        out.release()

    def release(self):
        """Release video capture resources."""
        if self.cap:
            self.cap.release()

    def __del__(self):
        """Cleanup on deletion."""
        self.release()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
