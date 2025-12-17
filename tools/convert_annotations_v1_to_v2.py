#!/usr/bin/env python3
"""
Convert Sweep-Annotator annotations from v1 format to v2 format.

Usage:
    python convert_annotations_v1_to_v2.py <dataset_path>

Example:
    python convert_annotations_v1_to_v2.py data/sweep2E_dualarm_v1

Description:
    This script converts annotation files from v1 format (coordinates normalized
    to [0, 1] as floats) to v2 format (coordinates normalized to [0, 1000] as integers).

    The original annotations.json file will be backed up as annotations_v1.json,
    and a new annotations.json file with v2 format will be created.

    Coordinate Conversion:
        - v1: [0, 1] range with floating point numbers (e.g., 0.549, 0.446)
        - v2: [0, 1000] range with integers (e.g., 549, 446)

    The conversion formula is: coord_v2 = round(coord_v1 * 1000)
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any


def convert_coordinate_v1_to_v2(coord: float) -> int:
    """
    Convert a single coordinate from v1 format [0, 1] to v2 format [0, 1000].

    Args:
        coord: Coordinate in v1 format (float in range [0, 1])

    Returns:
        Coordinate in v2 format (integer in range [0, 1000])
    """
    return int(round(coord * 1000))


def convert_coordinate_list_v1_to_v2(coord_list: List[float]) -> List[int]:
    """
    Convert a coordinate pair from v1 to v2 format.

    Args:
        coord_list: [x, y] in v1 format

    Returns:
        [x, y] in v2 format
    """
    return [convert_coordinate_v1_to_v2(coord_list[0]),
            convert_coordinate_v1_to_v2(coord_list[1])]


def convert_primitive_string(primitive_string: str) -> str:
    """
    Convert primitive string from v1 format to v2 format.

    Args:
        primitive_string: Primitive string in v1 format (e.g., "<Sweep> <Box> <0.549, 0.446, ...>")

    Returns:
        Primitive string in v2 format (e.g., "<Sweep> <Box> <549, 446, ...>")
    """
    import re

    # Find all float numbers in the string
    def replace_float(match):
        float_val = float(match.group(0))
        return str(convert_coordinate_v1_to_v2(float_val))

    # Replace all floating point numbers with integers
    converted = re.sub(r'\d+\.\d+', replace_float, primitive_string)

    return converted


def convert_primitive_v1_to_v2(primitive: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a primitive annotation from v1 to v2 format.

    Args:
        primitive: Primitive dictionary in v1 format

    Returns:
        Primitive dictionary in v2 format
    """
    converted_primitive = primitive.copy()

    # Convert coordinates
    if 'coordinates' in primitive and primitive['coordinates']:
        converted_primitive['coordinates'] = [
            convert_coordinate_list_v1_to_v2(coord)
            for coord in primitive['coordinates']
        ]

    # Convert target_position if present
    if 'target_position' in primitive and primitive['target_position'] is not None:
        converted_primitive['target_position'] = convert_coordinate_list_v1_to_v2(
            primitive['target_position']
        )

    # Convert primitive_string
    if 'primitive_string' in primitive:
        converted_primitive['primitive_string'] = convert_primitive_string(
            primitive['primitive_string']
        )

    return converted_primitive


def convert_segment_v1_to_v2(segment: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a segment from v1 to v2 format.

    Args:
        segment: Segment dictionary in v1 format

    Returns:
        Segment dictionary in v2 format (without overlap_next)
    """
    converted_segment = {
        'episode_id': segment['episode_id'],
        'start_frame': segment['start_frame'],
        'end_frame': segment['end_frame'],
        'primitive': convert_primitive_v1_to_v2(segment['primitive'])
    }

    return converted_segment


def convert_annotations_v1_to_v2(annotations_v1: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert entire annotations file from v1 to v2 format.

    Args:
        annotations_v1: Annotations dictionary in v1 format

    Returns:
        Annotations dictionary in v2 format
    """
    segments_v2 = [
        convert_segment_v1_to_v2(segment)
        for segment in annotations_v1['segments']
    ]

    return {
        'segments': segments_v2,
        'total_segments': len(segments_v2)
    }


def main():
    """Main conversion function."""
    if len(sys.argv) != 2:
        print("Error: Dataset path is required")
        print("\nUsage:")
        print("    python convert_annotations_v1_to_v2.py <dataset_path>")
        print("\nExample:")
        print("    python convert_annotations_v1_to_v2.py data/sweep2E_dualarm_v1")
        sys.exit(1)

    dataset_path = Path(sys.argv[1])

    # Check if dataset path exists
    if not dataset_path.exists():
        print(f"Error: Dataset path does not exist: {dataset_path}")
        sys.exit(1)

    if not dataset_path.is_dir():
        print(f"Error: Dataset path is not a directory: {dataset_path}")
        sys.exit(1)

    # Define file paths
    annotations_file = dataset_path / "annotations.json"
    annotations_v1_backup = dataset_path / "annotations_v1.json"

    # Check if annotations.json exists
    if not annotations_file.exists():
        print(f"Error: annotations.json not found in {dataset_path}")
        sys.exit(1)

    # Check if backup already exists
    if annotations_v1_backup.exists():
        print(f"Warning: {annotations_v1_backup} already exists.")
        response = input("Do you want to overwrite it? (y/n): ")
        if response.lower() != 'y':
            print("Conversion cancelled.")
            sys.exit(0)

    print(f"Converting annotations from v1 to v2 format...")
    print(f"Dataset path: {dataset_path}")

    # Load v1 annotations
    try:
        with open(annotations_file, 'r', encoding='utf-8') as f:
            annotations_v1 = json.load(f)
        print(f"✓ Loaded annotations.json (v1 format)")
        print(f"  Total segments: {annotations_v1.get('total_segments', len(annotations_v1.get('segments', [])))}")
    except Exception as e:
        print(f"Error loading annotations.json: {e}")
        sys.exit(1)

    # Convert to v2
    try:
        annotations_v2 = convert_annotations_v1_to_v2(annotations_v1)
        print(f"✓ Converted to v2 format")
    except Exception as e:
        print(f"Error converting annotations: {e}")
        sys.exit(1)

    # Backup original file
    try:
        with open(annotations_v1_backup, 'w', encoding='utf-8') as f:
            json.dump(annotations_v1, f, indent=2, ensure_ascii=False)
        print(f"✓ Backed up original file to: {annotations_v1_backup.name}")
    except Exception as e:
        print(f"Error creating backup: {e}")
        sys.exit(1)

    # Save v2 annotations
    try:
        with open(annotations_file, 'w', encoding='utf-8') as f:
            json.dump(annotations_v2, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved v2 annotations to: {annotations_file.name}")
    except Exception as e:
        print(f"Error saving v2 annotations: {e}")
        print(f"Restoring original file...")
        # Restore from backup
        with open(annotations_v1_backup, 'r', encoding='utf-8') as f:
            annotations_v1 = json.load(f)
        with open(annotations_file, 'w', encoding='utf-8') as f:
            json.dump(annotations_v1, f, indent=2, ensure_ascii=False)
        print(f"Original file restored")
        sys.exit(1)

    print("\n" + "="*60)
    print("Conversion completed successfully!")
    print("="*60)
    print(f"\nFiles:")
    print(f"  - Original (v1): {annotations_v1_backup}")
    print(f"  - Converted (v2): {annotations_file}")
    print(f"\nChanges:")
    print(f"  - Coordinates: [0, 1] floats → [0, 1000] integers")
    print(f"  - Removed: 'overlap_next' field")
    print(f"  - Total segments: {annotations_v2['total_segments']}")


if __name__ == "__main__":
    main()
