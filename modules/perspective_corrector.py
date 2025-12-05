"""Perspective correction module for camera view transformation."""
from typing import Tuple, Optional
import cv2
import numpy as np
import json
from datetime import datetime
from config.primitives_config import Coordinate


class PerspectiveCorrector:
    """Apply perspective transformation to correct tilted camera view."""

    def __init__(self, output_size: Tuple[int, int] = (224, 224)):
        """
        Initialize perspective corrector.

        Args:
            output_size: Output image size (width, height), default (224, 224)
        """
        self.output_size = output_size
        self.transform_matrix = None
        self.inverse_matrix = None
        self.src_points = None  # 4 points defining source quadrilateral
        self.dst_points = None  # 4 points defining destination rectangle

    def set_correction_points(self, src_points: np.ndarray, dst_points: Optional[np.ndarray] = None):
        """
        Define perspective transformation.

        Args:
            src_points: Source quadrilateral (4x2 array) - points in tilted image
            dst_points: Destination rectangle (4x2 array) - corrected positions.
                       If None, auto-generates rectangular bounds based on output_size.
        """
        if src_points.shape != (4, 2):
            raise ValueError(f"src_points must be (4, 2) array, got {src_points.shape}")

        self.src_points = src_points.astype(np.float32)

        # Auto-generate destination points if not provided
        if dst_points is None:
            width, height = self.output_size
            dst_points = np.array([
                [0, 0],  # Top-left
                [width - 1, 0],  # Top-right
                [width - 1, height - 1],  # Bottom-right
                [0, height - 1]  # Bottom-left
            ], dtype=np.float32)
        else:
            if dst_points.shape != (4, 2):
                raise ValueError(f"dst_points must be (4, 2) array, got {dst_points.shape}")
            dst_points = dst_points.astype(np.float32)

        self.dst_points = dst_points

        # Compute transformation matrices
        self.transform_matrix = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
        self.inverse_matrix = cv2.getPerspectiveTransform(self.dst_points, self.src_points)

    def correct_image(self, image: np.ndarray, output_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Apply perspective correction to image.

        Args:
            image: Input RGB image
            output_size: (width, height) of corrected image. If None, uses self.output_size

        Returns:
            Corrected RGB image
        """
        if self.transform_matrix is None:
            return image

        if output_size is None:
            output_size = self.output_size

        return cv2.warpPerspective(image, self.transform_matrix, output_size)

    def transform_point(self, point: Coordinate, inverse: bool = False) -> Coordinate:
        """
        Transform point coordinates.

        Args:
            point: Input coordinate
            inverse: If True, transform from corrected to original space

        Returns:
            Transformed coordinate
        """
        if self.transform_matrix is None:
            return point

        # Select appropriate matrix
        matrix = self.inverse_matrix if inverse else self.transform_matrix

        # Create homogeneous coordinate
        point_array = np.array([[[point.x, point.y]]], dtype=np.float32)

        # Apply transformation
        transformed = cv2.perspectiveTransform(point_array, matrix)

        return Coordinate(x=float(transformed[0, 0, 0]), y=float(transformed[0, 0, 1]))

    def save_calibration(self, filepath: str, dataset_name: str = ""):
        """
        Save calibration parameters to JSON.

        Args:
            filepath: Path to save calibration file
            dataset_name: Name of dataset for reference
        """
        if self.src_points is None or self.dst_points is None:
            raise ValueError("No calibration points set")

        calibration_data = {
            "src_points": self.src_points.tolist(),
            "dst_points": self.dst_points.tolist(),
            "output_size": list(self.output_size),
            "created_at": datetime.now().isoformat(),
            "dataset_name": dataset_name
        }

        with open(filepath, 'w') as f:
            json.dump(calibration_data, f, indent=2)

    def load_calibration(self, filepath: str):
        """
        Load calibration parameters from JSON.

        Args:
            filepath: Path to calibration file
        """
        with open(filepath, 'r') as f:
            calibration_data = json.load(f)

        src_points = np.array(calibration_data["src_points"], dtype=np.float32)
        dst_points = np.array(calibration_data["dst_points"], dtype=np.float32)
        self.output_size = tuple(calibration_data["output_size"])

        self.set_correction_points(src_points, dst_points)

    def is_calibrated(self) -> bool:
        """Check if calibration is set."""
        return self.transform_matrix is not None
