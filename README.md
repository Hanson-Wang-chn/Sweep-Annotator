# Trajectory Annotation Tool for Bimanual Manipulation Data

A web-based visualization and annotation tool for manually segmenting robot trajectories into primitive actions with geometric parameters. This tool processes LeRobot format datasets and exports annotated segments with structured primitive labels.

## Features

- **LeRobot Format Support**: Load and export data in LeRobot format (parquet + mp4)
- **Perspective Correction**: Apply perspective transformation to correct tilted camera views
- **Interactive Annotation**: Point-and-click interface for specifying geometric coordinates
- **Trajectory Segmentation**: Manually segment long trajectories into atomic action primitives
- **Multiple Primitive Types**: Support for sweep, clear, and refine operations
- **Multi-Camera Support**: Handle main camera and multiple wrist cameras
- **Overlap Management**: Configure overlapping frames between consecutive segments
- **Export Pipeline**: Generate new LeRobot datasets with primitive annotations

## Supported Primitive Types

1. **Sweep Box**: Define a box region and target position for sweeping motion
   - Requires: 2 corner points + 1 target position

2. **Sweep Triangle**: Define a triangular region and target position
   - Requires: 3 vertex points + 1 target position

3. **Clear Box**: Define a box region for clearing operations
   - Requires: 2 corner points

4. **Refine Line**: Define a line for refining operations
   - Requires: 2 endpoint coordinates

5. **Refine Arc**: Define an arc for refining operations
   - Requires: 3 control points

## Installation

### Requirements

- Python 3.10+
- LeRobot dataset in standard format

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd Sweep-Annotator
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
trajectory_annotation_tool/
├── main.py                      # Gradio application entry point
├── modules/
│   ├── data_loader.py          # LeRobot data loading
│   ├── video_processor.py      # Video frame extraction and processing
│   ├── perspective_corrector.py # Camera perspective correction
│   ├── trajectory_segmenter.py # Trajectory segmentation logic
│   ├── primitive_annotator.py  # Primitive annotation and coordinate capture
│   ├── data_exporter.py        # LeRobot format export
│   └── visualization.py        # Rendering utilities
├── config/
│   └── primitives_config.py    # Primitive type definitions
├── utils/
│   ├── coordinate_utils.py     # Coordinate transformation utilities
│   └── video_utils.py          # Video I/O utilities
└── requirements.txt
```

## Usage

### 1. Start the Application

```bash
python main.py
```

The Gradio interface will launch at `http://0.0.0.0:7860`

### 2. Load Dataset

1. Enter the path to your LeRobot dataset in the "Dataset Path" field
2. Click "Load Dataset"
3. Select an episode from the dropdown menu

### 3. Calibrate Perspective Correction (Optional)

If your main camera view is tilted:

1. Click "Set Calibration Points"
2. Click 4 points on the **Original View** image in this order:
   - Top-left corner of workspace
   - Top-right corner
   - Bottom-right corner
   - Bottom-left corner
3. The corrected view will automatically update
4. Click "Save Calibration" to save for future use
5. Use "Load Calibration" to reuse saved calibration

### 4. Annotate Segments

For each primitive action segment:

1. Navigate to the start frame using the frame slider
2. Select the primitive type from the dropdown
3. Click "Mark Start Frame"
4. Navigate to the end frame
5. Click on the **Corrected View** image to specify required coordinates:
   - Follow the on-screen instructions for point count
   - For sweep operations, specify the region first, then the target position
6. (Optional) Set overlap frames if the next segment should overlap
7. Click "Mark End & Save Segment"
8. Repeat for all segments across all episodes

### 5. Manage Segments

- View all annotated segments in the segment list
- Click on a segment row to delete it
- Review primitive strings and frame ranges

### 6. Export Annotated Dataset

1. Enter the output path for the new dataset
2. Click "Export Annotated Dataset"
3. The tool will:
   - Apply perspective correction to main camera videos
   - Export video segments for all cameras
   - Generate new parquet files with reset timestamps
   - Create metadata with primitive annotations
   - Compute dataset statistics

## Output Format

The exported dataset follows LeRobot format with additional primitive annotations:

### Directory Structure
```
output_dataset/
├── data/
│   └── chunk-000/
│       └── episode_XXXXXX.parquet
├── meta/
│   ├── info.json
│   ├── episodes.jsonl          # Includes primitive_annotation field
│   ├── episodes_stats.jsonl
│   └── tasks.jsonl
└── videos/
    └── chunk-000/
        ├── observation.images.main/
        ├── observation.images.secondary_0/
        └── observation.images.secondary_1/
```

### Episode Metadata with Primitive Annotation

```json
{
  "episode_index": 0,
  "tasks": ["sweep_to_shapes"],
  "primitive_annotation": {
    "type": "sweep_box",
    "coordinates": [[0.3, 0.4], [0.6, 0.7]],
    "target_position": [0.8, 0.5],
    "primitive_string": "<Sweep> <Box> <0.300, 0.400, 0.600, 0.700> <to> <Position> <0.800, 0.500>"
  },
  "source_episode": 0,
  "source_frame_start": 0,
  "source_frame_end": 150
}
```

## Key Technical Details

### Coordinate System

- All coordinates are normalized to [0, 1] relative to the **corrected image** dimensions
- Main camera: Coordinates apply to perspective-corrected view (default 224x224)
- Wrist cameras: No perspective correction applied
- Coordinate origin: Top-left corner (x increases right, y increases down)

### Video Processing

- **Main Camera**: Each frame undergoes perspective correction and re-encoding
- **Wrist Cameras**: Frames are extracted and re-encoded without transformation
- Output videos use high bitrate to maintain quality
- All videos maintain original FPS (typically 10 fps)

### Data Statistics

The tool automatically recomputes action and state statistics for the new dataset, as segmentation changes data distribution. Statistics are saved to `meta/episodes_stats.jsonl`.

### Overlapping Segments

Adjacent segments can overlap by specifying overlap frames. Overlapping data is physically duplicated in the exported dataset, allowing each episode to be independent.

## Example Workflow

```bash
# 1. Start the tool
python main.py

# 2. In the web interface:
#    - Load dataset: /path/to/sweep2E_dualarm_v1
#    - Load episode 0
#    - Set calibration (if needed)
#    - Mark start frame: 0
#    - Select primitive: "Sweep Box"
#    - Click 2 corners of box on corrected view
#    - Click 1 target position
#    - Mark end frame: 150
#    - Save segment
#    - Repeat for all primitives
#    - Export to: /path/to/output_dataset

# 3. The new dataset is ready for training
```

## Example Dataset

The repository includes a sample dataset (`sweep2E_dualarm_v1/`) with:
- 15 episodes of bimanual manipulation trajectories
- 3 camera views (main + 2 wrist cameras)
- 14-DOF action space (7 DOF per arm)
- 16-DOF state observation (8 DOF per arm with gripper)

## Development

### Running Tests

```bash
# Unit tests
pytest tests/

# Integration tests
pytest tests/integration/
```

### Module Overview

- **data_loader.py**: Parses LeRobot metadata and loads episodes
- **video_processor.py**: OpenCV-based video frame extraction
- **perspective_corrector.py**: Perspective transformation using cv2.warpPerspective
- **trajectory_segmenter.py**: Manages segment state and operations
- **primitive_annotator.py**: Coordinate capture and validation
- **data_exporter.py**: Multi-process video export and metadata generation
- **visualization.py**: Annotation overlays (boxes, triangles, lines, arcs)

## Troubleshooting

### Issue: Video frames appear black
- Check video codec compatibility (av1, h264, h265 supported)
- Verify video files are not corrupted
- Try converting videos to h264: `ffmpeg -i input.mp4 -c:v libx264 output.mp4`

### Issue: Calibration points don't align
- Ensure you click points in correct order: top-left, top-right, bottom-right, bottom-left
- Click on clearly visible workspace corners
- Reload calibration if needed

### Issue: Export fails with memory error
- Close other applications to free memory
- Process fewer segments at once
- Reduce video quality in data_exporter.py if needed

### Issue: Coordinates seem incorrect
- Remember: coordinates are normalized to corrected image size (224x224 by default)
- Check that perspective correction was applied before annotation
- Verify you're clicking on the corrected view, not original view

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{trajectory_annotation_tool,
  title={Trajectory Annotation Tool for Bimanual Manipulation},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/Sweep-Annotator}
}
```

## License

[Add your license here]

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with clear description

## Contact

For questions or issues, please open a GitHub issue or contact [your-email].

## Acknowledgments

- Built with [Gradio](https://gradio.app/) for the web interface
- Uses [LeRobot](https://github.com/huggingface/lerobot) data format
- Perspective correction powered by OpenCV
