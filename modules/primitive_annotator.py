"""Primitive annotation module for interactive coordinate capture."""
from typing import List, Tuple, Optional
import cv2
import numpy as np
from config.primitives_config import PrimitiveType, Coordinate


class PrimitiveAnnotator:
    """Handle interactive primitive annotation with coordinate capture."""

    def __init__(self, primitive_type: PrimitiveType):
        """
        Initialize primitive annotator.

        Args:
            primitive_type: Type of primitive to annotate
        """
        self.primitive_type = primitive_type
        self.coordinates: List[Coordinate] = []
        self.target_position: Optional[Coordinate] = None

    def get_required_points(self) -> Tuple[int, bool]:
        """
        Returns number of points needed for this primitive type.

        Returns:
            Tuple of (number_of_points, needs_target)
            - SWEEP_BOX: 2 points (box corners) + 1 target
            - SWEEP_TRIANGLE: 3 points (triangle vertices) + 1 target
            - CLEAR_BOX: 2 points (box corners)
            - REFINE_LINE: 2 points (line endpoints)
            - REFINE_ARC: 3 points (arc control points)
        """
        mapping = {
            PrimitiveType.SWEEP_BOX: (2, True),
            PrimitiveType.SWEEP_TRIANGLE: (3, True),
            PrimitiveType.CLEAR_BOX: (2, False),
            PrimitiveType.REFINE_LINE: (2, False),
            PrimitiveType.REFINE_ARC: (3, False)
        }
        return mapping[self.primitive_type]

    def add_coordinate(self, x: float, y: float, is_target: bool = False):
        """
        Add clicked coordinate.

        Args:
            x, y: Normalized coordinates [0, 1] in corrected image
            is_target: Whether this is target position for sweep
        """
        coord = Coordinate(x, y)
        if is_target:
            self.target_position = coord
        else:
            self.coordinates.append(coord)

    def is_complete(self) -> bool:
        """Check if all required points are captured."""
        required_points, needs_target = self.get_required_points()
        has_coords = len(self.coordinates) >= required_points
        has_target = (not needs_target) or (self.target_position is not None)
        return has_coords and has_target

    def undo_last_point(self):
        """Remove last added coordinate."""
        required_points, needs_target = self.get_required_points()

        # If target is set and we have all coordinates, remove target first
        if needs_target and self.target_position is not None and len(self.coordinates) >= required_points:
            self.target_position = None
        elif self.coordinates:
            self.coordinates.pop()

    def reset(self):
        """Clear all coordinates."""
        self.coordinates = []
        self.target_position = None

    def to_primitive_string(self) -> str:
        """
        Generate primitive string representation.

        Returns:
            String like "<Sweep> <Box> <x1, y1, x2, y2> <to> <Position> <x4, y4>"
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

    def visualize_annotation(self, image: np.ndarray, image_size: Tuple[int, int]) -> np.ndarray:
        """
        Draw annotation overlay on image.

        Args:
            image: RGB image
            image_size: (width, height) of the image for coordinate scaling

        Returns:
            Image with annotation overlay (boxes, triangles, lines, etc.)
        """
        overlay = image.copy()
        width, height = image_size

        if len(self.coordinates) == 0:
            return overlay

        # Convert normalized coordinates to pixel coordinates
        pixel_coords = [(int(c.x * width), int(c.y * height)) for c in self.coordinates]

        # Draw based on primitive type
        if self.primitive_type == PrimitiveType.SWEEP_BOX or self.primitive_type == PrimitiveType.CLEAR_BOX:
            if len(pixel_coords) >= 1:
                # Draw first point
                cv2.circle(overlay, pixel_coords[0], 5, (0, 255, 0), -1)
            if len(pixel_coords) >= 2:
                # Draw rectangle
                cv2.rectangle(overlay, pixel_coords[0], pixel_coords[1], (0, 255, 0), 2)
                cv2.circle(overlay, pixel_coords[1], 5, (0, 255, 0), -1)

        elif self.primitive_type == PrimitiveType.SWEEP_TRIANGLE:
            # Draw points
            for i, pt in enumerate(pixel_coords):
                cv2.circle(overlay, pt, 5, (0, 255, 0), -1)
                if i > 0:
                    cv2.line(overlay, pixel_coords[i - 1], pt, (0, 255, 0), 2)
            if len(pixel_coords) == 3:
                cv2.line(overlay, pixel_coords[2], pixel_coords[0], (0, 255, 0), 2)

        elif self.primitive_type == PrimitiveType.REFINE_LINE:
            if len(pixel_coords) >= 1:
                cv2.circle(overlay, pixel_coords[0], 5, (255, 0, 0), -1)
            if len(pixel_coords) >= 2:
                cv2.line(overlay, pixel_coords[0], pixel_coords[1], (255, 0, 0), 2)
                cv2.circle(overlay, pixel_coords[1], 5, (255, 0, 0), -1)

        elif self.primitive_type == PrimitiveType.REFINE_ARC:
            # Draw points
            for pt in pixel_coords:
                cv2.circle(overlay, pt, 5, (255, 0, 0), -1)
            # Draw lines connecting points
            if len(pixel_coords) >= 2:
                cv2.line(overlay, pixel_coords[0], pixel_coords[1], (255, 0, 0), 2)
            if len(pixel_coords) >= 3:
                cv2.line(overlay, pixel_coords[1], pixel_coords[2], (255, 0, 0), 2)

        # Draw target position if applicable
        if self.target_position:
            target_pixel = (int(self.target_position.x * width), int(self.target_position.y * height))
            cv2.circle(overlay, target_pixel, 8, (255, 0, 255), 2)
            cv2.circle(overlay, target_pixel, 3, (255, 0, 255), -1)
            # Draw arrow from shape to target
            if pixel_coords:
                center = tuple(np.mean(pixel_coords, axis=0).astype(int))
                cv2.arrowedLine(overlay, center, target_pixel, (255, 0, 255), 2, tipLength=0.3)

        return overlay

    def get_status_string(self) -> str:
        """
        Get current annotation status as string.

        Returns:
            Status string describing what's captured and what's needed
        """
        required_points, needs_target = self.get_required_points()

        status_parts = []
        status_parts.append(f"Captured {len(self.coordinates)}/{required_points} points")

        if needs_target:
            target_status = "✓" if self.target_position else "✗"
            status_parts.append(f"Target: {target_status}")

        if self.is_complete():
            status_parts.append("COMPLETE")
        else:
            if len(self.coordinates) < required_points:
                status_parts.append(f"Need {required_points - len(self.coordinates)} more point(s)")
            elif needs_target and not self.target_position:
                status_parts.append("Need target position")

        return " | ".join(status_parts)
