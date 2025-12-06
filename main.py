"""Main Gradio application for trajectory annotation."""
import gradio as gr
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import json


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


from modules.data_loader import LeRobotDataLoader
from modules.video_processor import VideoProcessor
from modules.perspective_corrector import PerspectiveCorrector
from modules.trajectory_segmenter import TrajectorySegmenter
from modules.primitive_annotator import PrimitiveAnnotator
from modules.data_exporter import LeRobotExporter
from modules.visualization import add_text_overlay
from config.primitives_config import PrimitiveType, PrimitiveAnnotation, Coordinate, TrajectorySegment
from utils.coordinate_utils import pixel_to_normalized


class AnnotationApp:
    """Main application state manager."""

    def __init__(self):
        self.data_loader: Optional[LeRobotDataLoader] = None
        self.current_episode = None
        self.video_processors: Dict[str, VideoProcessor] = {}
        self.perspective_corrector = PerspectiveCorrector(output_size=(224, 224))
        self.trajectory_segmenter: Optional[TrajectorySegmenter] = None
        self.primitive_annotator: Optional[PrimitiveAnnotator] = None
        self.calibration_mode = False
        self.calibration_points = []
        self.all_segments = []  # Segments from all episodes

        # Episode-specific data cache
        self.episode_data_cache: Dict[int, pd.DataFrame] = {}
        self.episode_video_processors_cache: Dict[int, Dict[str, VideoProcessor]] = {}


app = AnnotationApp()


def load_dataset(dataset_path: str) -> Tuple[gr.Dropdown, str]:
    """Load dataset and populate episode dropdown."""
    try:
        app.data_loader = LeRobotDataLoader(dataset_path)
        app.data_loader.load_dataset_metadata()
        episode_list = app.data_loader.get_episode_list()

        return (
            gr.Dropdown(choices=episode_list, value=episode_list[0] if episode_list else None),
            f"✓ Loaded dataset with {len(episode_list)} episodes"
        )
    except Exception as e:
        return gr.Dropdown(choices=[]), f"✗ Error loading dataset: {str(e)}"


def load_episode(episode_id: int) -> Tuple[gr.Slider, str, np.ndarray, np.ndarray]:
    """Load episode and initialize video processors."""
    if app.data_loader is None or episode_id is None:
        return gr.Slider(), "No dataset loaded", None, None

    try:
        # Load episode
        app.current_episode = app.data_loader.load_episode(episode_id)

        # Initialize video processors
        app.video_processors = {}
        for camera_name, video_path in app.current_episode.video_paths.items():
            app.video_processors[camera_name] = VideoProcessor(video_path)

        # Cache episode data and processors
        app.episode_data_cache[episode_id] = app.current_episode.data
        app.episode_video_processors_cache[episode_id] = app.video_processors

        # Initialize trajectory segmenter
        app.trajectory_segmenter = TrajectorySegmenter(app.current_episode)

        # Load first frame
        original_frame = app.video_processors["main"].get_frame(0)
        corrected_frame = app.perspective_corrector.correct_image(original_frame) if app.perspective_corrector.is_calibrated() else original_frame

        return (
            gr.Slider(minimum=0, maximum=app.current_episode.total_frames - 1, value=0, step=1),
            f"✓ Loaded episode {episode_id} ({app.current_episode.total_frames} frames)",
            original_frame,
            corrected_frame
        )
    except Exception as e:
        return gr.Slider(), f"✗ Error loading episode: {str(e)}", None, None


def update_frame(episode_id: int, frame_idx: int) -> Tuple[np.ndarray, np.ndarray, str]:
    """Update video displays for current frame."""
    if not app.video_processors or "main" not in app.video_processors:
        return None, None, "No episode loaded"

    try:
        frame_idx = int(frame_idx)
        original_frame = app.video_processors["main"].get_frame(frame_idx)

        if original_frame is None:
            return None, None, f"Failed to load frame {frame_idx}"

        # Apply perspective correction if calibrated
        if app.perspective_corrector.is_calibrated():
            corrected_frame = app.perspective_corrector.correct_image(original_frame)
        else:
            corrected_frame = original_frame.copy()

        # Add annotation overlay if annotator is active
        if app.primitive_annotator:
            corrected_frame = app.primitive_annotator.visualize_annotation(
                corrected_frame, app.perspective_corrector.output_size
            )

        # Add frame info
        original_frame = add_text_overlay(original_frame, f"Frame: {frame_idx}", (10, 30))
        corrected_frame = add_text_overlay(corrected_frame, f"Frame: {frame_idx}", (10, 30))

        return original_frame, corrected_frame
    except Exception as e:
        return None, None, f"Error: {str(e)}"


def enter_calibration_mode() -> str:
    """Enter calibration mode."""
    app.calibration_mode = True
    app.calibration_points = []
    return "📍 Calibration mode active. Click 4 points on the ORIGINAL image (top-left, top-right, bottom-right, bottom-left)"


def handle_image_click(frame_idx: int, primitive_type_str: str, evt: gr.SelectData) -> Tuple[str, str, np.ndarray]:
    """Handle click on corrected image."""
    if not app.video_processors or "main" not in app.video_processors:
        return "No episode loaded", "", None

    x, y = evt.index[0], evt.index[1]

    # Handle calibration mode
    if app.calibration_mode:
        app.calibration_points.append((x, y))

        if len(app.calibration_points) == 4:
            # Set calibration with 4 points
            src_points = np.array(app.calibration_points, dtype=np.float32)
            app.perspective_corrector.set_correction_points(src_points)
            app.calibration_mode = False
            app.calibration_points = []

            # Update display
            original_frame = app.video_processors["main"].get_frame(int(frame_idx))
            corrected_frame = app.perspective_corrector.correct_image(original_frame)
            corrected_frame = add_text_overlay(corrected_frame, f"Frame: {int(frame_idx)}", (10, 30))

            return "✓ Calibration complete!", "", corrected_frame
        else:
            return f"📍 Calibration: {len(app.calibration_points)}/4 points captured", "", None

    # Handle annotation mode
    if app.primitive_annotator is None:
        return "No annotation in progress", "", None

    # Convert pixel to normalized coordinates
    coord = pixel_to_normalized(x, y, app.perspective_corrector.output_size)

    # Check if this should be target position
    required_points, needs_target = app.primitive_annotator.get_required_points()
    is_target = needs_target and len(app.primitive_annotator.coordinates) >= required_points

    app.primitive_annotator.add_coordinate(coord.x, coord.y, is_target=is_target)

    # Update display
    original_frame = app.video_processors["main"].get_frame(int(frame_idx))
    corrected_frame = app.perspective_corrector.correct_image(original_frame)
    corrected_frame = app.primitive_annotator.visualize_annotation(
        corrected_frame, app.perspective_corrector.output_size
    )
    corrected_frame = add_text_overlay(corrected_frame, f"Frame: {int(frame_idx)}", (10, 30))

    status = app.primitive_annotator.get_status_string()
    coords_text = app.primitive_annotator.to_primitive_string() if app.primitive_annotator.is_complete() else "Incomplete"

    return status, coords_text, corrected_frame


def start_annotation(frame_idx: int, primitive_type_str: str) -> str:
    """Start new segment annotation."""
    if app.trajectory_segmenter is None:
        return "✗ No episode loaded"

    try:
        app.trajectory_segmenter.start_segment(int(frame_idx))

        # Initialize primitive annotator
        primitive_type = {
            "Sweep Box": PrimitiveType.SWEEP_BOX,
            "Sweep Triangle": PrimitiveType.SWEEP_TRIANGLE,
            "Clear Box": PrimitiveType.CLEAR_BOX,
            "Refine Line": PrimitiveType.REFINE_LINE,
            "Refine Arc": PrimitiveType.REFINE_ARC
        }[primitive_type_str]

        app.primitive_annotator = PrimitiveAnnotator(primitive_type)

        return f"✓ Started segment at frame {int(frame_idx)}. Click on corrected image to annotate."
    except Exception as e:
        return f"✗ Error: {str(e)}"


def end_annotation(frame_idx: int, overlap_frames: int) -> Tuple[pd.DataFrame, str]:
    """Complete segment annotation."""
    if app.trajectory_segmenter is None or app.primitive_annotator is None:
        return None, "✗ No annotation in progress"

    if not app.primitive_annotator.is_complete():
        return None, "✗ Annotation incomplete. Capture all required points first."

    try:
        # Get episode data for timestamps
        episode_data = app.current_episode.data
        start_frame = app.trajectory_segmenter.current_segment_start
        end_frame = int(frame_idx)

        start_timestamp = episode_data.iloc[start_frame]['timestamp']
        end_timestamp = episode_data.iloc[end_frame]['timestamp']

        # Create primitive annotation
        primitive = PrimitiveAnnotation(
            primitive_type=app.primitive_annotator.primitive_type,
            coordinates=app.primitive_annotator.coordinates.copy(),
            target_position=app.primitive_annotator.target_position,
            start_frame=start_frame,
            end_frame=end_frame,
            episode_id=app.current_episode.episode_id,
            timestamp_start=start_timestamp,
            timestamp_end=end_timestamp
        )

        # End segment
        segment = app.trajectory_segmenter.end_segment(end_frame, primitive, int(overlap_frames))
        app.all_segments.append(segment)

        # Reset annotator
        app.primitive_annotator = None

        # Update segments dataframe
        segments_data = []
        for seg in app.all_segments:
            segments_data.append({
                "Episode": seg.episode_id,
                "Start Frame": seg.start_frame,
                "End Frame": seg.end_frame,
                "Primitive": seg.primitive.primitive_type.value,
                "String": seg.primitive.to_string()
            })

        df = pd.DataFrame(segments_data)

        return df, f"✓ Segment saved! Total segments: {len(app.all_segments)}"
    except Exception as e:
        return None, f"✗ Error: {str(e)}"


def reset_annotation_points() -> Tuple[str, str]:
    """Reset annotation points."""
    if app.primitive_annotator:
        app.primitive_annotator.reset()
        return "Points reset", ""
    return "No annotation in progress", ""


def undo_last_point() -> Tuple[str, str]:
    """Undo last annotation point."""
    if app.primitive_annotator:
        app.primitive_annotator.undo_last_point()
        status = app.primitive_annotator.get_status_string()
        coords_text = app.primitive_annotator.to_primitive_string() if app.primitive_annotator.is_complete() else "Incomplete"
        return status, coords_text
    return "No annotation in progress", ""


def save_calibration(dataset_path: str) -> str:
    """Save calibration to file."""
    if not app.perspective_corrector.is_calibrated():
        return "✗ No calibration to save"

    try:
        calibration_path = Path(dataset_path) / "calibration.json"
        app.perspective_corrector.save_calibration(str(calibration_path), dataset_name=Path(dataset_path).name)
        return f"✓ Calibration saved to {calibration_path}"
    except Exception as e:
        return f"✗ Error saving calibration: {str(e)}"


def load_calibration_file(dataset_path: str) -> Tuple[str, np.ndarray, np.ndarray]:
    """Load calibration from file."""
    try:
        calibration_path = Path(dataset_path) / "calibration.json"
        app.perspective_corrector.load_calibration(str(calibration_path))

        # Update display if episode is loaded
        if app.video_processors and "main" in app.video_processors:
            original_frame = app.video_processors["main"].get_frame(0)
            corrected_frame = app.perspective_corrector.correct_image(original_frame)
            return f"✓ Calibration loaded from {calibration_path}", original_frame, corrected_frame

        return f"✓ Calibration loaded from {calibration_path}", None, None
    except Exception as e:
        return f"✗ Error loading calibration: {str(e)}", None, None


def save_annotations(dataset_path: str) -> Tuple[pd.DataFrame, str]:
    """Save all annotations to file."""
    if not app.all_segments:
        return None, "✗ No annotations to save"

    if not dataset_path:
        return None, "✗ Please specify dataset path"

    try:
        annotations_path = Path(dataset_path) / "annotations.json"

        # Convert all segments to dictionaries
        annotations_data = {
            "segments": [seg.to_dict() for seg in app.all_segments],
            "total_segments": len(app.all_segments)
        }

        # Save to JSON file
        with open(annotations_path, 'w') as f:
            json.dump(annotations_data, f, indent=2, cls=NumpyEncoder)

        return None, f"✓ Saved {len(app.all_segments)} annotations to {annotations_path}"
    except Exception as e:
        return None, f"✗ Error saving annotations: {str(e)}"


def load_annotations(dataset_path: str) -> Tuple[pd.DataFrame, str]:
    """Load annotations from file."""
    if not dataset_path:
        return None, "✗ Please specify dataset path"

    try:
        annotations_path = Path(dataset_path) / "annotations.json"

        if not annotations_path.exists():
            return None, f"✗ No annotations file found at {annotations_path}"

        # Load JSON file
        with open(annotations_path, 'r') as f:
            annotations_data = json.load(f)

        # Convert dictionaries back to TrajectorySegment objects
        app.all_segments = [
            TrajectorySegment.from_dict(seg_data)
            for seg_data in annotations_data["segments"]
        ]

        # Update segments dataframe
        segments_data = []
        for seg in app.all_segments:
            segments_data.append({
                "Episode": seg.episode_id,
                "Start Frame": seg.start_frame,
                "End Frame": seg.end_frame,
                "Primitive": seg.primitive.primitive_type.value,
                "String": seg.primitive.to_string()
            })

        df = pd.DataFrame(segments_data)

        return df, f"✓ Loaded {len(app.all_segments)} annotations from {annotations_path}"
    except Exception as e:
        return None, f"✗ Error loading annotations: {str(e)}"


def export_dataset(output_path: str) -> str:
    """Export annotated dataset."""
    if not app.all_segments:
        return "✗ No segments to export"

    if not output_path:
        return "✗ Please specify output path"

    try:
        # Create exporter
        exporter = LeRobotExporter(output_path, str(app.data_loader.dataset_path))

        # Export all segments
        def progress_callback(current, total, message):
            print(f"[{current}/{total}] {message}")

        exporter.export_all_segments(
            app.all_segments,
            app.episode_data_cache,
            app.episode_video_processors_cache,
            app.perspective_corrector,
            progress_callback=progress_callback
        )

        return f"✓ Successfully exported {len(app.all_segments)} segments to {output_path}"
    except Exception as e:
        return f"✗ Error exporting dataset: {str(e)}"


def delete_segment(segments_df: pd.DataFrame, evt: gr.SelectData) -> Tuple[pd.DataFrame, str]:
    """Delete selected segment."""
    if evt is None or segments_df is None or len(segments_df) == 0:
        return segments_df, "No segment selected"

    try:
        row_idx = evt.index[0]
        if 0 <= row_idx < len(app.all_segments):
            deleted_seg = app.all_segments[row_idx]
            app.all_segments.pop(row_idx)

            # Update dataframe
            segments_data = []
            for seg in app.all_segments:
                segments_data.append({
                    "Episode": seg.episode_id,
                    "Start Frame": seg.start_frame,
                    "End Frame": seg.end_frame,
                    "Primitive": seg.primitive.primitive_type.value,
                    "String": seg.primitive.to_string()
                })

            df = pd.DataFrame(segments_data) if segments_data else pd.DataFrame(columns=["Episode", "Start Frame", "End Frame", "Primitive", "String"])

            return df, f"✓ Deleted segment (Episode {deleted_seg.episode_id}, frames {deleted_seg.start_frame}-{deleted_seg.end_frame})"

        return segments_df, "Invalid segment index"
    except Exception as e:
        return segments_df, f"✗ Error: {str(e)}"


# Build Gradio UI
with gr.Blocks(title="Sweep Annotator") as demo:
    gr.Markdown("# Sweep Annotator")

    with gr.Row():
        dataset_path_input = gr.Textbox(label="Dataset Path", placeholder="/path/to/lerobot/dataset", scale=3)
        load_dataset_btn = gr.Button("Load Dataset", scale=1)

    dataset_status = gr.Textbox(label="Status", interactive=False)

    with gr.Row():
        episode_dropdown = gr.Dropdown(label="Episode", choices=[], interactive=True, scale=1)
        frame_slider = gr.Slider(minimum=0, maximum=10000, step=1, label="Frame", scale=4)

    with gr.Row():
        with gr.Column():
            original_video_display = gr.Image(label="Original View (Click to calibrate)", type="numpy", interactive=True) # 可以添加参数 height=400 调整高度，但图片不会自动缩放
        with gr.Column():
            corrected_video_display = gr.Image(label="Corrected View (Click to annotate)", type="numpy", interactive=True)

    gr.Markdown("## Segmentation & Annotation")
    
    with gr.Row():
        primitive_type_dropdown = gr.Dropdown(
            label="Primitive Type",
            choices=["Sweep Box", "Sweep Triangle", "Clear Box", "Refine Line", "Refine Arc"],
            value="Sweep Box",
            scale=1
        )
        annotation_status = gr.Textbox(label="Annotation Status", interactive=False, scale=1)
        coordinates_display = gr.Textbox(label="Captured Coordinates", interactive=False, scale=1)

    with gr.Row():
        start_frame_btn = gr.Button("Mark Start Frame", variant="primary")
        overlap_frames_input = gr.Number(label="Overlap Frames", value=0, precision=0)
        end_frame_btn = gr.Button("Mark End Frame & Save Segment", variant="primary")

    with gr.Row():
        reset_points_btn = gr.Button("Reset Points")
        undo_point_btn = gr.Button("Undo Last Point")

    gr.Markdown("## Segment List")
    gr.Markdown("**WARNING: Click to delete.**")
    segments_dataframe = gr.Dataframe(
        headers=["Episode", "Start Frame", "End Frame", "Primitive", "String"],
        datatype=["number", "number", "number", "str", "str"],
        interactive=False
    )

    delete_segment_status = gr.Textbox(label="Delete Status", interactive=False)

    gr.Markdown("## Save/Load Annotations")
    gr.Markdown("Save your annotation progress to resume later, or load previously saved annotations.")
    with gr.Row():
        save_annotations_btn = gr.Button("Save Annotations", variant="primary")
        load_annotations_btn = gr.Button("Load Annotations", variant="primary")

    annotations_status = gr.Textbox(label="Annotations Status", interactive=False)

    gr.Markdown("## Perspective Correction")
    with gr.Row():
        calibration_mode_btn = gr.Button("Set Calibration Points")
        load_calibration_btn = gr.Button("Load Calibration")
        save_calibration_btn = gr.Button("Save Calibration")

    calibration_status = gr.Textbox(label="Calibration Status", interactive=False)

    gr.Markdown("## Export")
    with gr.Row():
        output_path_input = gr.Textbox(label="Output Path", placeholder="/path/to/output/dataset", scale=3)
        export_btn = gr.Button("Export Annotated Dataset", variant="primary", scale=1)

    export_status = gr.Textbox(label="Export Status", interactive=False)

    # Event handlers
    load_dataset_btn.click(
        fn=load_dataset,
        inputs=[dataset_path_input],
        outputs=[episode_dropdown, dataset_status]
    )

    episode_dropdown.change(
        fn=load_episode,
        inputs=[episode_dropdown],
        outputs=[frame_slider, dataset_status, original_video_display, corrected_video_display]
    )

    frame_slider.change(
        fn=update_frame,
        inputs=[episode_dropdown, frame_slider],
        outputs=[original_video_display, corrected_video_display]
    )

    calibration_mode_btn.click(
        fn=enter_calibration_mode,
        outputs=[calibration_status]
    )

    original_video_display.select(
        fn=handle_image_click,
        inputs=[frame_slider, primitive_type_dropdown],
        outputs=[calibration_status, coordinates_display, corrected_video_display]
    )

    corrected_video_display.select(
        fn=handle_image_click,
        inputs=[frame_slider, primitive_type_dropdown],
        outputs=[annotation_status, coordinates_display, corrected_video_display]
    )

    start_frame_btn.click(
        fn=start_annotation,
        inputs=[frame_slider, primitive_type_dropdown],
        outputs=[annotation_status]
    )

    end_frame_btn.click(
        fn=end_annotation,
        inputs=[frame_slider, overlap_frames_input],
        outputs=[segments_dataframe, annotation_status]
    )

    reset_points_btn.click(
        fn=reset_annotation_points,
        outputs=[annotation_status, coordinates_display]
    )

    undo_point_btn.click(
        fn=undo_last_point,
        outputs=[annotation_status, coordinates_display]
    )

    save_calibration_btn.click(
        fn=save_calibration,
        inputs=[dataset_path_input],
        outputs=[calibration_status]
    )

    load_calibration_btn.click(
        fn=load_calibration_file,
        inputs=[dataset_path_input],
        outputs=[calibration_status, original_video_display, corrected_video_display]
    )

    export_btn.click(
        fn=export_dataset,
        inputs=[output_path_input],
        outputs=[export_status]
    )

    segments_dataframe.select(
        fn=delete_segment,
        inputs=[segments_dataframe],
        outputs=[segments_dataframe, delete_segment_status]
    )

    save_annotations_btn.click(
        fn=save_annotations,
        inputs=[dataset_path_input],
        outputs=[segments_dataframe, annotations_status]
    )

    load_annotations_btn.click(
        fn=load_annotations,
        inputs=[dataset_path_input],
        outputs=[segments_dataframe, annotations_status]
    )


if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
