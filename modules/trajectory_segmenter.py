"""Trajectory segmentation module for managing segment state."""
from typing import List, Optional
from config.primitives_config import LeRobotEpisode, TrajectorySegment, PrimitiveAnnotation


class TrajectorySegmenter:
    """Manage trajectory segmentation state and operations."""

    def __init__(self, episode: LeRobotEpisode):
        """
        Initialize trajectory segmenter.

        Args:
            episode: Episode to segment
        """
        self.episode = episode
        self.segments: List[TrajectorySegment] = []
        self.current_segment_start: Optional[int] = None

    def start_segment(self, frame_idx: int):
        """
        Mark start of new segment.

        Args:
            frame_idx: Frame index to start segment
        """
        if frame_idx < 0 or frame_idx >= self.episode.total_frames:
            raise ValueError(f"Frame index {frame_idx} out of range [0, {self.episode.total_frames})")

        self.current_segment_start = frame_idx

    def end_segment(self, frame_idx: int, primitive: PrimitiveAnnotation) -> TrajectorySegment:
        """
        Complete current segment.

        Args:
            frame_idx: End frame index
            primitive: Annotated primitive for this segment

        Returns:
            Created TrajectorySegment
        """
        if self.current_segment_start is None:
            raise ValueError("No segment started. Call start_segment() first.")

        if frame_idx < self.current_segment_start:
            raise ValueError(f"End frame {frame_idx} must be >= start frame {self.current_segment_start}")

        if frame_idx >= self.episode.total_frames:
            raise ValueError(f"End frame {frame_idx} out of range [0, {self.episode.total_frames})")

        segment = TrajectorySegment(
            episode_id=self.episode.episode_id,
            start_frame=self.current_segment_start,
            end_frame=frame_idx,
            primitive=primitive
        )

        self.segments.append(segment)
        self.current_segment_start = None

        return segment

    def delete_segment(self, segment_idx: int):
        """
        Remove segment by index.

        Args:
            segment_idx: Index of segment to delete
        """
        if segment_idx < 0 or segment_idx >= len(self.segments):
            raise ValueError(f"Segment index {segment_idx} out of range [0, {len(self.segments)})")

        del self.segments[segment_idx]

    def get_segments(self) -> List[TrajectorySegment]:
        """
        Return all segments for this episode.

        Returns:
            List of trajectory segments
        """
        return self.segments

    def get_segment_by_index(self, idx: int) -> Optional[TrajectorySegment]:
        """
        Get segment by index.

        Args:
            idx: Segment index

        Returns:
            TrajectorySegment or None if not found
        """
        if 0 <= idx < len(self.segments):
            return self.segments[idx]
        return None

    def is_segment_in_progress(self) -> bool:
        """Check if a segment is currently being defined."""
        return self.current_segment_start is not None

    def cancel_current_segment(self):
        """Cancel the current segment in progress."""
        self.current_segment_start = None

    def get_total_segments(self) -> int:
        """Get total number of segments."""
        return len(self.segments)
