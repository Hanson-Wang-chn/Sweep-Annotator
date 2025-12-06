# Sweep Annotator

A Gradio-based GUI tool for annotating robot manipulation trajectories with primitive actions. This tool enables efficient annotation of sweeping, clearing, and refinement primitives on LeRobot format datasets with perspective correction capabilities.

## Features

- Load/export annotated datasets in LeRobot format
- Interactive GUI for trajectory annotation
- Support for multiple primitive types (Sweep Box, Sweep Triangle, Clear Box, Refine Line, Refine Arc)
- Perspective correction/calibration for top-down view
- Segment-based annotation with frame ranges
- Save/load annotation progress
- Visual feedback during annotation process

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd Sweep-Annotator
```

2. Create virtual environment:
```bash
conda create -n sweep-annotator python=3.11 -y
conda activate sweep-annotator
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Launch the application:
```bash
python main.py
```

The GUI will be available at `http://localhost:7860`.

## Dataset Format

This tool expects datasets in LeRobot format with the following structure:

```
dataset_path/
├── meta/
│   ├── info.json           # Dataset metadata (fps, camera names, etc.)
│   └── episodes.jsonl      # Episode metadata
├── data/
│   └── chunk-*/
│       └── episode_*.parquet
└── videos/
    └── chunk-*/
        └── observation.images.*/
            └── episode_*.mp4
```

## Usage Guide

### 1. Loading a Dataset

1. **Enter Dataset Path**: In the "Dataset Path" field, enter the full path to your LeRobot dataset directory
   - Example: `/path/to/lerobot/dataset`

2. **Click "Load Dataset"**: The tool will:
   - Load dataset metadata
   - Populate the episode dropdown with available episodes
   - Display status message indicating success or errors

3. **Select Episode**: Choose an episode from the dropdown menu
   - The first frame will automatically load in both Original and Corrected views
   - Frame slider will be configured based on episode length

### 2. Perspective Calibration

Calibration is recommended for accurate top-down annotations. This step corrects the camera perspective to create a normalized view.

**Steps:**

1. **Navigate to First Frame**: Use the frame slider to go to frame 0 (or any frame with clear view of the workspace)

2. **Click "Set Calibration Points"**: This enters calibration mode
   - Status will show: "Calibration mode active. Click 4 points on the ORIGINAL image..."

3. **Click 4 Points on Original Image** (left side):
   - **Point 1**: Top-left corner of workspace
   - **Point 2**: Top-right corner of workspace
   - **Point 3**: Bottom-right corner of workspace
   - **Point 4**: Bottom-left corner of workspace

   **The order matters!** Always go: top-left → top-right → bottom-right → bottom-left

4. **Auto-Apply**: After the 4th point, calibration automatically applies
   - Corrected view updates to show the perspective-corrected image
   - Status shows: "Calibration complete!"

5. **Save/Load Calibration** (Optional): Click "Save Calibration" to save calibration points
   - Saves to `<dataset_path>/calibration.json`
   - Can be reloaded later with "Load Calibration"

**Calibration Tips:**
- Choose a frame with clear workspace boundaries
- Points should form a quadrilateral around your workspace
- Once calibrated, all frames will use the same correction
- Corrected view will be 224x224 pixels

### 3. Annotating Trajectories

#### Understanding Segments

A **segment** is a portion of a trajectory (range of frames) with a single primitive annotation. Each segment contains:
- Start frame and end frame
- Primitive type and coordinates
- Optional target position
- Optional overlap with next segment

#### Annotation Workflow

**Step 1: Select Primitive Type**
- Choose from the "Primitive Type" dropdown:
  - **Sweep Box**: Sweeping action within a box (4 corner points + target)
  - **Sweep Triangle**: Sweeping within a triangle (3 corner points + target)
  - **Clear Box**: Clearing action within a box (4 corner points, no target)
  - **Refine Line**: Refinement along a line (2 endpoints, no target)
  - **Refine Arc**: Refinement along an arc (3 points defining the arc, no target)

**Step 2: Mark Start Frame**
1. Move frame slider to the starting frame of the segment
2. Click "Mark Start Frame"
3. Status will show: "Started segment at frame X"

**Step 3: Click Points on Corrected Image** (right side)
- The number of points depends on primitive type:
  - **Box**: 4 corner points (top-left, top-right, bottom-right, bottom-left)
  - **Triangle**: 3 corner points
  - **Line**: 2 endpoints
  - **Arc**: 3 points defining the arc

- **Visual Feedback**:
  - Red circles show clicked points
  - Green circle shows target position (for Sweep primitives)
  - Lines connect the points to show the shape

- **Status Display**: Shows progress like "Captured 2/4 points"

- **Coordinates Display**: Shows captured coordinates in normalized format [0-1]

**Step 4: Add Target Position** (Sweep primitives only)
- After capturing all shape points, click one more time for the target position
- This indicates where the swept material should end up
- Status will show: "Captured 4/4 points + target position"

**Step 5: Mark End Frame**
1. Move frame slider to the ending frame of the segment
2. (Optional) Set "Overlap Frames" if this segment overlaps with the next
   - Example: If frames 90-100 of this segment overlap with the next, enter 10
3. Click "Mark End Frame & Save Segment"

**Step 6: Segment Saved**
- The segment appears in the "Segment List" table
- Shows: Episode, Start Frame, End Frame, Primitive type, and Coordinate string
- Annotation status resets for the next segment

#### Annotation Controls

- **Reset Points**: Clear all clicked points and start over (keeps start frame)
- **Undo Last Point**: Remove the most recent point clicked
- **Click on Corrected Image**: Add annotation points
- **Frame Slider**: Navigate through frames (doesn't affect annotation)

#### Coordinate Format

Coordinates are stored in normalized format [0-1] and displayed as:
```
<Primitive> <Shape> <x1, y1, x2, y2, ...> <to> <Position> <xt, yt>
```

Example:
```
<Sweep> <Box> <0.100, 0.200, 0.300, 0.200, 0.300, 0.400, 0.100, 0.400> <to> <Position> <0.500, 0.500>
```

### 4. Managing Segments

#### Segment List Table

The segment list shows all annotated segments across all episodes:

| Episode | Start Frame | End Frame | Primitive | String |
|---------|-------------|-----------|-----------|--------|
| 0 | 10 | 50 | sweep_box | \<Sweep\> \<Box\> ... |
| 0 | 45 | 80 | clear_box | \<Clear\> \<Box\> ... |

#### Deleting Segments

**WARNING**: Clicking on any row in the segment list will DELETE that segment!

- Click any row to delete it
- This action cannot be undone (unless you reload saved annotations)

### 5. Interrupt and Resume

You can save your progress and resume annotation later.

#### Saving Annotations

1. Click "Save Annotations"
2. Annotations save to `<dataset_path>/annotations.json`
3. Status shows: "Saved X annotations to ..."

**What's saved:**
- All segment data (frames, primitive types, coordinates)
- Episode IDs and timestamps
- Target positions

#### Loading Annotations

1. Click "Load Annotations"
2. Loads from `<dataset_path>/annotations.json`
3. All segments appear in the Segment List
4. Status shows: "Loaded X annotations from ..."

**Workflow for Resuming:**
1. Launch the application
2. Load the dataset
3. Load calibration (if previously saved)
4. Load annotations
5. Continue annotating new segments

### 6. Exporting Annotated Dataset

After completing annotations, export the processed dataset.

**Steps:**

1. **Enter Output Path**: Specify where to save the exported dataset
   - Example: `/path/to/output/annotated_dataset`

2. **Click "Export Annotated Dataset"**

**What happens during export:**
- Creates new LeRobot format dataset at output path
- Splits episodes into segments based on annotations
- Applies perspective correction to all frames
- Copies and processes robot state/action data
- Creates new episode files for each segment

**Export Output Structure:**
```
output_path/
├── meta/
│   ├── info.json
│   └── episodes.jsonl       # One episode per segment
├── data/
│   └── chunk-*/
│       └── episode_*.parquet  # State/action data per segment
└── videos/
    └── chunk-*/
        └── observation.images.*/
            └── episode_*.mp4    # Corrected video per segment
```

**Console Output**: Progress messages show which segments are being processed:
```
[1/10] Processing segment Episode 0, frames 10-50
[2/10] Processing segment Episode 0, frames 45-80
...
```

### 7. GUI Information Display

The interface provides real-time information:

#### Status Fields

- **Status** (top): Dataset loading status, episode loading status
- **Calibration Status**: Calibration progress and confirmation
- **Annotation Status**: Point capture progress, segment saving status
- **Captured Coordinates**: Real-time coordinate display
- **Delete Status**: Segment deletion confirmation
- **Annotations Status**: Save/load operation results
- **Export Status**: Export progress and completion

#### Visual Information

**Original View (Left)**:
- Shows raw camera feed
- Used for calibration point selection
- Displays frame number overlay
- Click here only during calibration

**Corrected View (Right)**:
- Shows perspective-corrected image (224x224)
- Used for annotation point selection
- Shows visual annotation overlay:
  - **Red circles**: Captured shape points
  - **Green circle**: Target position
  - **Lines**: Connect points to show shape
- Displays frame number overlay
- Click here for annotations

**Frame Slider**:
- Shows current frame number
- Range: 0 to (total_frames - 1)
- Can freely navigate without affecting annotation

**Segment List**:
- Real-time updates as segments are added
- Shows all episodes and their segments
- Click to delete (be careful!)

## Tips and Best Practices

1. **Calibrate Once**: Save calibration and reuse it for the entire dataset if camera doesn't move

2. **Consistent Point Order**: Always click points in consistent order (e.g., top-left, top-right, bottom-right, bottom-left for boxes)

3. **Frame Selection**: Choose start/end frames where the action clearly begins/ends

4. **Save Frequently**: Use "Save Annotations" regularly to avoid losing work

5. **Overlap Frames**: Use overlap when segments have transitional frames that belong to both

6. **Visual Verification**: After clicking points, verify the visualization matches your intent

7. **Segment Deletion**: Be careful when clicking the segment list table - it deletes immediately!

## Troubleshooting

### Dataset won't load
- Verify the dataset path is correct
- Check that `meta/info.json` exists
- Ensure LeRobot format is correct

### Calibration points not working
- Make sure you click on the **Original** (left) image, not the Corrected one
- Click exactly 4 points in order
- Try again with "Set Calibration Points"

### Annotation points not appearing
- Ensure you clicked "Mark Start Frame" first
- Click on the **Corrected** (right) image, not the Original one
- Check that Annotation Status shows segment is active

### Export fails
- Verify output path is writable
- Check that all segments have valid frame ranges
- Ensure calibration is set

### Segments missing after reload
- Make sure you clicked "Save Annotations" before closing
- Check that `annotations.json` exists in dataset directory
- Verify the dataset path is correct when loading

## Data Format Reference

### Primitive Types

1. **Sweep Box**: `<Sweep> <Box> <x1,y1, x2,y2, x3,y3, x4,y4> <to> <Position> <xt,yt>`
2. **Sweep Triangle**: `<Sweep> <Triangle> <x1,y1, x2,y2, x3,y3> <to> <Position> <xt,yt>`
3. **Clear Box**: `<Clear> <Box> <x1,y1, x2,y2, x3,y3, x4,y4>`
4. **Refine Line**: `<Refine> <Line> <x1,y1, x2,y2>`
5. **Refine Arc**: `<Refine> <Arc> <x1,y1, x2,y2, x3,y3>`

All coordinates are normalized to [0, 1] range.
