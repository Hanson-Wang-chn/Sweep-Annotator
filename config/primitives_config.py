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
    x: int  # Normalized [0, 1000] coordinates (integers)
    y: int

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {"x": self.x, "y": self.y}

    def to_list(self) -> List[int]:
        """Convert to list [x, y]."""
        return [self.x, self.y]

    @classmethod
    def from_dict(cls, data: Dict) -> 'Coordinate':
        """Create from dictionary."""
        return cls(x=int(data["x"]), y=int(data["y"]))

    @classmethod
    def from_list(cls, data: List) -> 'Coordinate':
        """Create from list [x, y]."""
        return cls(x=int(data[0]), y=int(data[1]))


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

        # Format coordinates as integers
        coord_strs = [f"{c.x}, {c.y}" for c in self.coordinates]
        coord_part = f"<{', '.join(coord_strs)}>"

        if self.target_position:
            return f"<{primitive_name}> <{shape_name}> {coord_part} <to> <Position> <{self.target_position.x}, {self.target_position.y}>"
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

    @classmethod
    def from_dict(cls, data: Dict) -> 'PrimitiveAnnotation':
        """Create from dictionary for JSON deserialization."""
        return cls(
            primitive_type=PrimitiveType(data["type"]),
            coordinates=[Coordinate.from_list(c) for c in data["coordinates"]],
            target_position=Coordinate.from_list(data["target_position"]) if data["target_position"] else None,
            start_frame=data["start_frame"],
            end_frame=data["end_frame"],
            episode_id=data["episode_id"],
            timestamp_start=data["timestamp_start"],
            timestamp_end=data["timestamp_end"]
        )


@dataclass
class TrajectorySegment:
    """Represents an annotated segment of a trajectory."""
    episode_id: int
    start_frame: int
    end_frame: int
    primitive: PrimitiveAnnotation

    def get_frame_count(self) -> int:
        """Get total number of frames in this segment."""
        return self.end_frame - self.start_frame + 1

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "episode_id": self.episode_id,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "primitive": self.primitive.to_dict()
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'TrajectorySegment':
        """Create from dictionary for JSON deserialization."""
        return cls(
            episode_id=data["episode_id"],
            start_frame=data["start_frame"],
            end_frame=data["end_frame"],
            primitive=PrimitiveAnnotation.from_dict(data["primitive"])
        )


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
