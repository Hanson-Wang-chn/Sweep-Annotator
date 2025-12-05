"""Primitive type definitions and data structures."""
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from enum import Enum
import pandas as pd


class PrimitiveType(Enum):
    """Primitive action types for robot manipulation."""
    SWEEP_BOX = "sweep_box"
    SWEEP_TRIANGLE = "sweep_triangle"
    CLEAR_BOX = "clear_box"
    REFINE_LINE = "refine_line"
    REFINE_ARC = "refine_arc"


@dataclass
class Coordinate:
    """2D coordinate in image space."""
    x: float  # Normalized [0, 1] or pixel coordinates
    y: float

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {"x": self.x, "y": self.y}

    def to_list(self) -> List[float]:
        """Convert to list [x, y]."""
        return [self.x, self.y]

    @classmethod
    def from_dict(cls, data: Dict) -> 'Coordinate':
        """Create from dictionary."""
        return cls(x=data["x"], y=data["y"])

    @classmethod
    def from_list(cls, data: List[float]) -> 'Coordinate':
        """Create from list [x, y]."""
        return cls(x=data[0], y=data[1])


@dataclass
class PrimitiveAnnotation:
    """Structured primitive annotation data."""
    primitive_type: PrimitiveType
    coordinates: List[Coordinate]  # Number varies by type
    target_position: Optional[Coordinate]  # For sweep operations
    start_frame: int
    end_frame: int
    episode_id: int
    timestamp_start: float
    timestamp_end: float

    def to_string(self) -> str:
        """
        Convert to primitive string format.
        Example: <Sweep> <Box> <x1, y1, x2, y2> <to> <Position> <x4, y4>
        """
        primitive_name = self.primitive_type.value.split('_')[0].capitalize()
        shape_name = self.primitive_type.value.split('_')[1].capitalize() if '_' in self.primitive_type.value else ""

        # Format coordinates
        coord_strs = [f"{c.x:.3f}, {c.y:.3f}" for c in self.coordinates]
        coord_part = f"<{', '.join(coord_strs)}>"

        if self.target_position:
            return f"<{primitive_name}> <{shape_name}> {coord_part} <to> <Position> <{self.target_position.x:.3f}, {self.target_position.y:.3f}>"
        else:
            return f"<{primitive_name}> <{shape_name}> {coord_part}"

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "type": self.primitive_type.value,
            "coordinates": [c.to_list() for c in self.coordinates],
            "target_position": self.target_position.to_list() if self.target_position else None,
            "primitive_string": self.to_string(),
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "episode_id": self.episode_id,
            "timestamp_start": self.timestamp_start,
            "timestamp_end": self.timestamp_end
        }


@dataclass
class TrajectorySegment:
    """Represents an annotated segment of a trajectory."""
    episode_id: int
    start_frame: int
    end_frame: int
    primitive: PrimitiveAnnotation
    overlap_next: int  # Number of overlapping frames with next segment

    def get_frame_count(self) -> int:
        """Get total number of frames in this segment."""
        return self.end_frame - self.start_frame + 1

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "episode_id": self.episode_id,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "overlap_next": self.overlap_next,
            "primitive": self.primitive.to_dict()
        }


@dataclass
class LeRobotEpisode:
    """Represents a single episode in LeRobot format."""
    episode_id: int
    total_frames: int
    fps: int
    parquet_path: str
    video_paths: Dict[str, str]  # {"main": path, "secondary_0": path, "secondary_1": path}
    data: Optional[pd.DataFrame] = None  # Loaded parquet data

    def __post_init__(self):
        """Validate episode data."""
        if self.total_frames <= 0:
            raise ValueError(f"Invalid total_frames: {self.total_frames}")
        if self.fps <= 0:
            raise ValueError(f"Invalid fps: {self.fps}")
