# Trajectory Annotation Tool for Bimanual Manipulation Data
## Technical Specification Document

## 1. Project Overview

### 1.1 Purpose
Develop a visualization and annotation tool to manually segment long robot trajectories into primitive actions, annotate them with structured primitive labels, and apply perspective correction to camera images. The tool processes LeRobot format data and outputs annotated segments in the same format.

### 1.2 Key Requirements
- Load and visualize LeRobot format trajectory data (parquet + mp4)
- Apply perspective correction to tilted main camera view
- Manually segment long trajectories into atomic action clips
- Interactive annotation with point-and-click coordinate specification
- Generate structured primitive labels with geometric parameters
- Export annotated segments in LeRobot format
- Support overlapping segments between consecutive primitives

## 2. Technology Stack

### 2.1 Core Dependencies
- **Python**: 3.10+
- **Gradio**: 4.x - Web-based UI framework
- **OpenCV**: 4.x - Image processing and perspective transformation
- **PyArrow**: For parquet file handling
- **Pandas**: Data manipulation
- **NumPy**: Numerical operations
- **lerobot**: For LeRobot format I/O
- **Pillow**: Image handling for Gradio
- **opencv-python-headless**: Video processing

### 2.2 Additional Libraries
- **json**: Primitive structure serialization
- **pathlib**: File path management
- **datetime**: Timestamp handling
- **dataclasses**: Data structure definitions

## 3. System Architecture

### 3.1 Module Structure
```
trajectory_annotation_tool/
├── main.py                      # Gradio application entry point
├── modules/
│   ├── __init__.py
│   ├── data_loader.py          # LeRobot data loading
│   ├── video_processor.py      # Video frame extraction and processing
│   ├── perspective_corrector.py # Camera perspective correction
│   ├── trajectory_segmenter.py # Trajectory segmentation logic
│   ├── primitive_annotator.py  # Primitive annotation and coordinate capture
│   ├── data_exporter.py        # LeRobot format export
│   └── visualization.py        # Rendering utilities
├── config/
│   ├── __init__.py
│   └── primitives_config.py    # Primitive type definitions
├── utils/
│   ├── __init__.py
│   ├── coordinate_utils.py     # Coordinate transformation utilities
│   └── video_utils.py          # Video I/O utilities
└── requirements.txt
```

## 4. Data Structures

### 4.1 Primitive Definitions
```python
from dataclasses import dataclass
from typing import List, Tuple, Optional
from enum import Enum

class PrimitiveType(Enum):
    SWEEP_BOX = "sweep_box"
    SWEEP_TRIANGLE = "sweep_triangle"
    CLEAR_BOX = "clear_box"
    REFINE_LINE = "refine_line"
    REFINE_ARC = "refine_arc"

@dataclass
class Coordinate:
    x: float  # Normalized [0, 1] or pixel coordinates
    y: float
    
@dataclass
class PrimitiveAnnotation:
    primitive_type: PrimitiveType
    coordinates: List[Coordinate]  # Number varies by type
    target_position: Optional[Coordinate]  # For sweep operations
    start_frame: int
    end_frame: int
    episode_id: int
    timestamp_start: float
    timestamp_end: float
    
    def to_string(self) -> str:
        """Convert to primitive string format"""
        # Example: <Sweep> <Box> <x1, y1, x2, y2> <to> <Position> <x4, y4>
        pass

@dataclass
class TrajectorySegment:
    episode_id: int
    start_frame: int
    end_frame: int
    primitive: PrimitiveAnnotation
    overlap_next: int  # Number of overlapping frames with next segment
```

### 4.2 Dataset Structure
```python
@dataclass
class LeRobotEpisode:
    episode_id: int
    total_frames: int
    fps: int
    parquet_path: str
    video_paths: dict  # {"main": path, "secondary_0": path, "secondary_1": path}
    data: pd.DataFrame  # Loaded parquet data
```

## 5. Module Specifications

### 5.1 DataLoader Module (`data_loader.py`)

#### Purpose
Load LeRobot format datasets from disk

#### Key Functions
```python
class LeRobotDataLoader:
    def __init__(self, dataset_path: str):
        """
        Args:
            dataset_path: Root path to LeRobot dataset
        """
        self.dataset_path = Path(dataset_path)
        self.episodes = []
        
    def load_dataset_metadata(self) -> dict:
        """
        Load meta/info.json and meta/episodes.jsonl
        Returns:
            dict: Dataset metadata including FPS, camera names, action dimensions
        """
        pass
    
    def get_episode_list(self) -> List[int]:
        """
        Returns:
            List of episode IDs available in dataset
        """
        pass
    
    def load_episode(self, episode_id: int) -> LeRobotEpisode:
        """
        Load single episode data (parquet + video paths)
        Args:
            episode_id: Episode number
        Returns:
            LeRobotEpisode object
        """
        pass
    
    def get_episode_frame_count(self, episode_id: int) -> int:
        """
        Returns:
            Total number of frames in episode
        """
        pass
```

### 5.2 VideoProcessor Module (`video_processor.py`)

#### Purpose
Extract and process video frames from mp4 files

#### Key Functions
```python
class VideoProcessor:
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
    def get_frame(self, frame_idx: int) -> np.ndarray:
        """
        Extract single frame at index
        Args:
            frame_idx: Frame index (0-based)
        Returns:
            RGB image as numpy array (H, W, 3)
        """
        pass
    
    def get_frame_range(self, start_frame: int, end_frame: int) -> List[np.ndarray]:
        """
        Extract range of frames
        Args:
            start_frame: Start frame index
            end_frame: End frame index (inclusive)
        Returns:
            List of RGB images
        """
        pass
    
    def export_segment(self, start_frame: int, end_frame: int, output_path: str):
        """
        Export video segment to new mp4 file
        Args:
            start_frame: Start frame index
            end_frame: End frame index
            output_path: Output video file path
        """
        pass
    
    def release(self):
        """Release video capture resources"""
        self.cap.release()
```

### 5.3 PerspectiveCorrector Module (`perspective_corrector.py`)

#### Purpose
Apply perspective transformation to correct tilted camera view

#### Key Functions
```python
class PerspectiveCorrector:
    def __init__(self):
    # 注意，需要能指定输出图片的尺寸，默认(224, 224)
        self.transform_matrix = None
        self.inverse_matrix = None
        self.src_points = None  # 4 points defining source quadrilateral
        self.dst_points = None  # 4 points defining destination rectangle
        
    def set_correction_points(self, src_points: np.ndarray, dst_points: np.ndarray):
        """
        Define perspective transformation
        Args:
            src_points: Source quadrilateral (4x2 array) - points in tilted image
            dst_points: Destination rectangle (4x2 array) - corrected positions
        """
        self.src_points = src_points
        self.dst_points = dst_points
        self.transform_matrix = cv2.getPerspectiveTransform(
            src_points.astype(np.float32), 
            dst_points.astype(np.float32)
        )
        self.inverse_matrix = cv2.getPerspectiveTransform(
            dst_points.astype(np.float32),
            src_points.astype(np.float32)
        )
        
    def correct_image(self, image: np.ndarray, output_size: Tuple[int, int]) -> np.ndarray:
        """
        Apply perspective correction to image
        Args:
            image: Input RGB image
            output_size: (width, height) of corrected image
        Returns:
            Corrected RGB image
        """
        if self.transform_matrix is None:
            return image
        return cv2.warpPerspective(image, self.transform_matrix, output_size)
    
    def transform_point(self, point: Coordinate, inverse: bool = False) -> Coordinate:
        """
        Transform point coordinates
        Args:
            point: Input coordinate
            inverse: If True, transform from corrected to original space
        Returns:
            Transformed coordinate
        """
        pass
    
    def save_calibration(self, filepath: str):
        """Save calibration parameters to JSON"""
        pass
    
    def load_calibration(self, filepath: str):
        """Load calibration parameters from JSON"""
        pass
```

### 5.4 TrajectorySegmenter Module (`trajectory_segmenter.py`)

#### Purpose
Manage trajectory segmentation state and operations

#### Key Functions
```python
class TrajectorySegmenter:
    def __init__(self, episode: LeRobotEpisode):
        self.episode = episode
        self.segments: List[TrajectorySegment] = []
        self.current_segment_start: Optional[int] = None
        
    def start_segment(self, frame_idx: int):
        """Mark start of new segment"""
        self.current_segment_start = frame_idx
        
    def end_segment(self, frame_idx: int, primitive: PrimitiveAnnotation, 
                   overlap_frames: int = 0) -> TrajectorySegment:
        """
        Complete current segment
        Args:
            frame_idx: End frame index
            primitive: Annotated primitive for this segment
            overlap_frames: Number of frames overlapping with next segment
        Returns:
            Created TrajectorySegment
        """
        segment = TrajectorySegment(
            episode_id=self.episode.episode_id,
            start_frame=self.current_segment_start,
            end_frame=frame_idx,
            primitive=primitive,
            overlap_next=overlap_frames
        )
        self.segments.append(segment)
        self.current_segment_start = None
        return segment
    
    def delete_segment(self, segment_idx: int):
        """Remove segment by index"""
        pass
    
    def get_segments(self) -> List[TrajectorySegment]:
        """Return all segments for this episode"""
        return self.segments
    
    def export_segment(self, segment: TrajectorySegment, 
                      episode_data: pd.DataFrame,
                      video_processor_map: dict, # 注意这里传入的是所有相机的processor
                      perspective_corrector: PerspectiveCorrector, # 新增传入矫正器
                      new_episode_id: int):
        """
        Export single segment as new episode with TRANSFORMATION.
        
        Logic:
        1. Main Camera: Read Frame -> Apply Perspective Correction -> Write to New Video (Re-encode).
        2. Wrist Cameras: Read Frame -> Write to New Video (Re-encode or Copy if strict trim needed).
        3. Parquet: Slice data rows -> Reset Timestamps -> Save.
        """
        pass
```

### 5.5 PrimitiveAnnotator Module (`primitive_annotator.py`)

#### Purpose
Handle interactive primitive annotation with coordinate capture

#### Key Functions
```python
class PrimitiveAnnotator:
    def __init__(self, primitive_type: PrimitiveType):
        self.primitive_type = primitive_type
        self.coordinates: List[Coordinate] = []
        self.target_position: Optional[Coordinate] = None
        
    def get_required_points(self) -> int:
        """
        Returns number of points needed for this primitive type
        - SWEEP_BOX: 2 points (box corners) + 1 target
        - SWEEP_TRIANGLE: 3 points (triangle vertices) + 1 target
        - CLEAR_BOX: 2 points (box corners)
        - REFINE_LINE: 2 points (line endpoints)
        - REFINE_ARC: 3 points (arc control points)
        """
        mapping = {
            PrimitiveType.SWEEP_BOX: (2, True),  # (points, needs_target)
            PrimitiveType.SWEEP_TRIANGLE: (3, True),
            PrimitiveType.CLEAR_BOX: (2, False),
            PrimitiveType.REFINE_LINE: (2, False),
            PrimitiveType.REFINE_ARC: (3, False)
        }
        return mapping[self.primitive_type]
    
    def add_coordinate(self, x: float, y: float, is_target: bool = False):
        """
        Add clicked coordinate
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
        """Check if all required points are captured"""
        required_points, needs_target = self.get_required_points()
        has_coords = len(self.coordinates) >= required_points
        has_target = (not needs_target) or (self.target_position is not None)
        return has_coords and has_target
    
    def reset(self):
        """Clear all coordinates"""
        self.coordinates = []
        self.target_position = None
    
    def to_primitive_string(self) -> str:
        """
        Generate primitive string representation
        Returns:
            String like "<Sweep> <Box> <x1, y1, x2, y2> <to> <Position> <x4, y4>"
        """
        pass
    
    def visualize_annotation(self, image: np.ndarray) -> np.ndarray:
        """
        Draw annotation overlay on image
        Args:
            image: RGB image
        Returns:
            Image with annotation overlay (boxes, triangles, lines, etc.)
        """
        pass
```

### 5.6 DataExporter Module (`data_exporter.py`)

#### Purpose
Export annotated segments in LeRobot format

#### Key Functions
```python
class LeRobotExporter:
    def __init__(self, output_path: str, original_dataset_path: str):
        """
        Args:
            output_path: Root directory for new dataset
            original_dataset_path: Path to original dataset for reference
        """
        self.output_path = Path(output_path)
        self.original_dataset_path = Path(original_dataset_path)
        
    def create_dataset_structure(self):
        """Create LeRobot directory structure (data/, meta/, videos/)"""
        pass
    
    def compute_and_save_stats(self):
        """
        CRITICAL FOR LEROBOT:
        Iterate over all new episodes, stack actions/states, 
        compute mean/std/min/max, and save to meta/episodes_stats.jsonl.
        Use lerobot.common.datasets.compute_stats logic.
        """
        pass
    
    def export_segment(self, segment: TrajectorySegment, 
                      episode_data: pd.DataFrame,
                      video_processor: VideoProcessor,
                      new_episode_id: int):
        """
        Export single segment as new episode
        Args:
            segment: Trajectory segment to export
            episode_data: Original episode parquet data
            video_processor: Video processor for frame extraction
            new_episode_id: ID for new episode in output dataset
        """
        # Extract relevant data rows
        # Export video segment for all cameras
        # Create parquet file
        # Update metadata
        pass
    
    def export_all_segments(self, segments: List[TrajectorySegment],
                          episode_data_map: dict,
                          video_processor_map: dict):
        """
        Export all segments as new dataset
        Args:
            segments: All annotated segments
            episode_data_map: {episode_id: DataFrame}
            video_processor_map: {episode_id: {camera_name: VideoProcessor}}
        """
        pass
    
    def create_metadata(self, segments: List[TrajectorySegment]):
        """
        Generate meta/info.json, meta/episodes.jsonl, meta/tasks.jsonl
        Args:
            segments: All exported segments
        """
        pass
    
    def add_primitive_annotations(self, segment: TrajectorySegment, 
                                 episode_metadata: dict):
        """
        Add primitive annotation to episode metadata
        Args:
            segment: Segment with primitive annotation
            episode_metadata: Episode entry in episodes.jsonl
        """
        pass
```

## 6. Gradio UI Specification

### 6.1 Application Layout
```
┌─────────────────────────────────────────────────────────────┐
│                     Trajectory Annotation Tool               │
├─────────────────────────────────────────────────────────────┤
│ Dataset Path: [___________________] [Load Dataset]           │
│ Episode: [Dropdown] | Frame: [Slider: 0 ─────●───── 1000]   │
├─────────────────────────────────────────────────────────────┤
│ ┌──────────────────────┐ ┌──────────────────────┐          │
│ │   Original View      │ │  Corrected View      │          │
│ │                      │ │                      │          │
│ │    [Main Camera]     │ │  [Click to annotate] │          │
│ │                      │ │                      │          │
│ └──────────────────────┘ └──────────────────────┘          │
├─────────────────────────────────────────────────────────────┤
│ Perspective Correction:                                      │
│ [Set Calibration Points] [Load Calibration] [Save]          │
├─────────────────────────────────────────────────────────────┤
│ Segmentation Controls:                                       │
│ Start Frame: [____] [Mark Start] | End Frame: [____]        │
│ Overlap Frames: [____] (default: 0)                         │
│ [Mark End & Annotate]                                        │
├─────────────────────────────────────────────────────────────┤
│ Primitive Annotation:                                        │
│ Type: [Dropdown: Sweep Box / Sweep Triangle / Clear Box / .]│
│ Coordinates: [List of captured points]                      │
│ [Reset Points] [Undo Last Point]                            │
│ Status: Captured X/Y points [Complete/Incomplete]           │
├─────────────────────────────────────────────────────────────┤
│ Segment List:                                                │
│ ┌──────────────────────────────────────────────────────┐   │
│ │ Ep 0 | Frame 0-150   | <Sweep><Box>...    [Edit][Del]│   │
│ │ Ep 0 | Frame 140-300 | <Clear><Box>...    [Edit][Del]│   │
│ │ ...                                                   │   │
│ └──────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│ Export:                                                      │
│ Output Path: [___________________]                           │
│ [Export Annotated Dataset]                                   │
│ Status: [_______________________________________________]    │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 Component Specifications

#### Input Components
```python
# Dataset loading
dataset_path_input = gr.Textbox(label="Dataset Path", placeholder="/path/to/lerobot/dataset")
load_dataset_btn = gr.Button("Load Dataset")

# Episode navigation
episode_dropdown = gr.Dropdown(label="Episode", choices=[], interactive=True)
frame_slider = gr.Slider(minimum=0, maximum=1000, step=1, label="Frame", interactive=True)

# Video display
original_video_display = gr.Image(label="Original View", type="numpy")
corrected_video_display = gr.Image(label="Corrected View (Click to annotate)", 
                                  type="numpy", interactive=True)

# Perspective correction
calibration_mode_btn = gr.Button("Set Calibration Points")
load_calibration_btn = gr.Button("Load Calibration")
save_calibration_btn = gr.Button("Save Calibration")
calibration_status = gr.Textbox(label="Calibration Status", interactive=False)

# Segmentation controls
start_frame_input = gr.Number(label="Start Frame", precision=0)
mark_start_btn = gr.Button("Mark Start")
end_frame_input = gr.Number(label="End Frame", precision=0)
overlap_frames_input = gr.Number(label="Overlap Frames", value=0, precision=0)
mark_end_btn = gr.Button("Mark End & Annotate")

# Primitive annotation
primitive_type_dropdown = gr.Dropdown(
    label="Primitive Type",
    choices=["Sweep Box", "Sweep Triangle", "Clear Box", "Refine Line", "Refine Arc"],
    value="Sweep Box"
)
coordinates_display = gr.Textbox(label="Captured Coordinates", interactive=False)
reset_points_btn = gr.Button("Reset Points")
undo_point_btn = gr.Button("Undo Last Point")
annotation_status = gr.Textbox(label="Annotation Status", interactive=False)

# Segment list
segments_dataframe = gr.Dataframe(
    headers=["Episode", "Start Frame", "End Frame", "Primitive", "Actions"],
    datatype=["number", "number", "number", "str", "str"],
    interactive=False
)

# Export
output_path_input = gr.Textbox(label="Output Path", placeholder="/path/to/output/dataset")
export_btn = gr.Button("Export Annotated Dataset")
export_status = gr.Textbox(label="Export Status", interactive=False)
```

### 6.3 Event Handlers
```python
def load_dataset(dataset_path: str) -> tuple:
    """
    Load dataset and populate episode dropdown
    Returns:
        (episode_choices, status_message)
    """
    pass

def update_frame_display(episode_id: int, frame_idx: int) -> tuple:
    """
    Update video displays for current frame
    Returns:
        (original_image, corrected_image)
    """
    pass

def on_calibration_mode():
    """Enter calibration mode - next 4 clicks define source points"""
    pass

def on_image_click(evt: gr.SelectData, primitive_type: str) -> tuple:
    """
    Handle click on corrected image
    Returns:
        (updated_coordinates_text, updated_annotation_status, updated_image_with_overlay)
    """
    pass

def mark_segment_start(frame_idx: int) -> str:
    """
    Mark start of new segment
    Returns:
        status_message
    """
    pass

def mark_segment_end(frame_idx: int, overlap: int, primitive_type: str) -> tuple:
    """
    Complete segment annotation
    Returns:
        (updated_segments_dataframe, status_message)
    """
    pass

def export_dataset(output_path: str) -> str:
    """
    Export all annotated segments
    Returns:
        status_message with progress
    """
    pass
```

### 6.4 State Management
```python
# Global state (using gr.State)
app_state = {
    "data_loader": None,  # LeRobotDataLoader instance
    "current_episode": None,  # LeRobotEpisode
    "video_processors": {},  # {camera_name: VideoProcessor}
    "perspective_corrector": None,  # PerspectiveCorrector instance
    "trajectory_segmenter": None,  # TrajectorySegmenter instance
    "primitive_annotator": None,  # PrimitiveAnnotator instance
    "calibration_mode": False,
    "calibration_points": [],  # Temporary list for calibration clicks
}

state = gr.State(value=app_state)
```

## 7. Workflow Specification

### 7.1 Initial Setup Workflow

1. User enters dataset path and clicks "Load Dataset"
2. System loads metadata and populates episode dropdown
3. User selects episode from dropdown
4. System loads first frame of episode

### 7.2 Calibration Workflow

1. User clicks "Set Calibration Points"
2. System enters calibration mode
3. User clicks 4 points on original image defining source quadrilateral
   - Top-left, top-right, bottom-right, bottom-left of workspace
4. System prompts for destination rectangle (or auto-generates rectangular bounds)
5. System computes and applies perspective transform
6. User can save calibration to JSON for reuse

### 7.3 Annotation Workflow

1. User navigates to desired start frame using slider
2. User clicks "Mark Start" to begin segment
3. User selects primitive type from dropdown
4. User navigates to end frame
5. User clicks on corrected image to specify coordinates:
   - For "Sweep Box": Click 2 corners, then 1 target position
   - For "Sweep Triangle": Click 3 vertices, then 1 target position
   - For "Clear Box": Click 2 corners
   - For "Refine Line": Click 2 endpoints
   - For "Refine Arc": Click 3 control points
6. System displays annotation overlay on image
7. User sets overlap frames if needed (default 0)
8. User clicks "Mark End & Annotate" to save segment
9. Repeat for all segments in episode
10. Repeat for all episodes

### 7.4 Export Workflow

1. User specifies output path
2. User clicks "Export Annotated Dataset"
3. System iterates through all segments:
   - Creates new episode for each segment
   - Extracts corresponding data from parquet
   - Exports video segments for all cameras
   - Generates metadata with primitive annotations
4. System displays progress and completion status

## 8. Data Format Specifications

### 8.1 LeRobot Input Format
```
dataset/
├── data/
│   └── chunk-000/
│       └── episode_XXXXXX.parquet  # Columns: timestamp, observation.*, action.*
├── meta/
│   ├── info.json                    # Dataset metadata
│   ├── episodes.jsonl               # Per-episode metadata
│   └── tasks.jsonl                  # Task descriptions
└── videos/
    └── chunk-000/
        └── observation.images.{camera_name}/
            └── episode_XXXXXX.mp4
```

### 8.2 Parquet Schema

Expected columns in episode parquet files:
- `frame_index`: int
- `timestamp`: float
- `observation.images.main`: (not stored, in video)
- `observation.images.secondary_0`: (not stored, in video)
- `observation.images.secondary_1`: (not stored, in video)
- `observation.state`: array (robot joint states)
- `action`: array (action values, e.g., 14-dim for bimanual)

### 8.3 Output LeRobot Format

Same structure as input, but:
- Each segment becomes a new episode
- `meta/episodes.jsonl` includes additional fields:
```json
  {
    "episode_index": 0,
    "tasks": ["sweep_to_shapes"],
    "primitive_annotation": {
      "type": "sweep_box",
      "coordinates": [[0.3, 0.4], [0.6, 0.7]],
      "target_position": [0.8, 0.5],
      "primitive_string": "<Sweep> <Box> <0.3, 0.4, 0.6, 0.7> <to> <Position> <0.8, 0.5>"
    },
    "source_episode": 0,
    "source_frame_start": 0,
    "source_frame_end": 150
  }
```

### 8.4 Calibration File Format
```json
{
  "src_points": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
  "dst_points": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
  "output_size": [width, height],
  "created_at": "2024-01-01T00:00:00",
  "dataset_name": "sweep2E_dualarm_v1"
}
```

## 9. Implementation Priority

### Phase 1: Core Functionality (Week 1)
1. Data loading module
2. Video processing module
3. Basic Gradio UI with frame navigation
4. Perspective correction module

### Phase 2: Annotation Features (Week 2)
1. Primitive annotator with coordinate capture
2. Trajectory segmenter
3. Interactive annotation workflow
4. Visualization overlays

### Phase 3: Export and Polish (Week 3)
1. Data exporter module
2. Segment list management (edit/delete)
3. Progress indicators
4. Error handling and validation
5. Documentation and testing

## 10. Error Handling

### 10.1 Input Validation
- Check dataset path exists and has correct structure
- Validate episode IDs before loading
- Ensure frame indices are within valid range
- Verify required number of points before completing annotation

### 10.2 User Feedback
- Display clear error messages for invalid inputs
- Show progress indicators for long operations (loading, exporting)
- Confirm before destructive operations (delete segment, overwrite dataset)
- Validate output path is writable before export

### 10.3 Edge Cases
- Handle missing video files gracefully
- Deal with corrupted parquet files
- Handle perspective correction failures
- Manage memory for large datasets (lazy loading)

## 11. Testing Requirements

### 11.1 Unit Tests
- Test each module independently
- Verify coordinate transformations
- Test primitive string generation
- Validate parquet/video I/O

### 11.2 Integration Tests
- Test full annotation workflow
- Verify exported dataset format
- Test calibration save/load
- Validate segment overlap handling

### 11.3 User Acceptance Tests
- Test with actual collected data (Z, E, N letters)
- Verify usability of UI
- Test export compatibility with LeRobot tools
- Validate coordinate accuracy

## 12. Performance Considerations

- Use lazy loading for videos (don't load all frames at once)
- Cache corrected frames for current episode
- Implement frame seeking optimization in VideoProcessor
- Use multiprocessing for parallel video export
- Display low-resolution preview while annotating, full-resolution for export

## 13. Future Enhancements (Post-MVP)

- Automatic segment suggestion using motion detection
- Keyboard shortcuts for faster annotation
- Batch processing multiple episodes
- Undo/redo for segment operations
- Preview mode to replay annotated segments
- Export to other formats (RLDS, etc.)
- Integration with VLM for automatic primitive suggestion

## 14. API Reference Summary

### Key Classes
- `LeRobotDataLoader`: Load and parse LeRobot datasets
- `VideoProcessor`: Extract and manipulate video frames
- `PerspectiveCorrector`: Apply geometric corrections
- `TrajectorySegmenter`: Manage segment state
- `PrimitiveAnnotator`: Capture and validate annotations
- `LeRobotExporter`: Generate output datasets

### Key Data Structures
- `LeRobotEpisode`: Represents single trajectory
- `TrajectorySegment`: Represents annotated segment
- `PrimitiveAnnotation`: Structured primitive data
- `Coordinate`: 2D point with transformations

### Main Workflow Functions
- `load_dataset()`: Initialize dataset
- `update_frame_display()`: Render current frame
- `on_image_click()`: Capture annotation points
- `mark_segment_end()`: Complete segment
- `export_dataset()`: Generate output

## 15. Dependencies Installation
```bash
pip install gradio>=4.0.0
pip install opencv-python-headless>=4.8.0
pip install pyarrow>=14.0.0
pip install pandas>=2.0.0
pip install numpy>=1.24.0
pip install pillow>=10.0.0
pip install lerobot  # Install from source if needed
```

## 16. Deliverables

1. Python package with all modules
2. Gradio web application (main.py)
3. Requirements.txt with pinned versions
4. README.md with usage instructions
5. Example calibration file
6. Unit tests for core modules

## 17. Notes

- Wrist Camera 不需要矫正，不需要仿射变换，直接裁剪时间段即可。需要在代码中区分相机名。
- 坐标归一化：确保Normalized [0, 1]是相对于 **矫正后(Transform applied)** 的图像尺寸归一化的。逻辑如下：
	- Gradio 显示矫正后的图
	- 用户点击
	- 记录坐标 (xclk​,yclk​)
	- 保存到 Json/Parquet 时，保存 (xclk​/Wnew​,yclk​/Hnew​)
	- **不需要**逆变换回原始图像，因为VLA训练也是在矫正图上进行的
- 在 `cv2.VideoWriter` 中明确指定高码率；在`export_all_segments`中使用多进程并行导出。
- 由于切分了数据，新的数据集分布可能与原始的大分布略有不同，LeRobot 的 Statistics必须重新计算
- 相邻小段可能有重叠，因此在导出为 LeRobot 格式（每个 Segment 变成一个独立的 Episode）时，数据在物理存储上是**复制**的。比如，Segment A (Frame 0-100), Segment B (Frame 90-150). Frame 90-100 的图像和 Action 会被分别存入 `episode_000` 和 `episode_001`。
