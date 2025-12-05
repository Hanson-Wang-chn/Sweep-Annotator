"""Data exporter module for LeRobot format output."""
from pathlib import Path
from typing import List, Dict, Optional
import json
import pandas as pd
import numpy as np
from datetime import datetime
from config.primitives_config import TrajectorySegment
from modules.video_processor import VideoProcessor
from modules.perspective_corrector import PerspectiveCorrector
from concurrent.futures import ProcessPoolExecutor, as_completed
import shutil


class LeRobotExporter:
    """Export annotated segments in LeRobot format."""

    def __init__(self, output_path: str, original_dataset_path: str):
        """
        Initialize exporter.

        Args:
            output_path: Root directory for new dataset
            original_dataset_path: Path to original dataset for reference
        """
        self.output_path = Path(output_path)
        self.original_dataset_path = Path(original_dataset_path)
        self.exported_episodes = []

        # Load original metadata
        with open(self.original_dataset_path / "meta" / "info.json", 'r') as f:
            self.original_metadata = json.load(f)

    def create_dataset_structure(self):
        """Create LeRobot directory structure (data/, meta/, videos/)."""
        self.output_path.mkdir(parents=True, exist_ok=True)
        (self.output_path / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
        (self.output_path / "meta").mkdir(parents=True, exist_ok=True)

        # Create video directories for each camera
        for feature_name in self.original_metadata["features"]:
            if feature_name.startswith("observation.images."):
                video_dir = self.output_path / "videos" / "chunk-000" / feature_name
                video_dir.mkdir(parents=True, exist_ok=True)

    def export_segment(self, segment: TrajectorySegment,
                      episode_data: pd.DataFrame,
                      video_processor_map: Dict[str, VideoProcessor],
                      perspective_corrector: PerspectiveCorrector,
                      new_episode_id: int) -> Dict:
        """
        Export single segment as new episode.

        Args:
            segment: Trajectory segment to export
            episode_data: Original episode parquet data
            video_processor_map: {camera_name: VideoProcessor}
            perspective_corrector: Perspective corrector for main camera
            new_episode_id: ID for new episode in output dataset

        Returns:
            Episode metadata dict
        """
        # Extract relevant data rows
        segment_data = episode_data[
            (episode_data['frame_index'] >= segment.start_frame) &
            (episode_data['frame_index'] <= segment.end_frame)
        ].copy()

        # Reset frame indices and timestamps
        segment_data['frame_index'] = range(len(segment_data))
        segment_data['episode_index'] = new_episode_id

        # Reset timestamps to start from 0
        if len(segment_data) > 0:
            first_timestamp = segment_data['timestamp'].iloc[0]
            segment_data['timestamp'] = segment_data['timestamp'] - first_timestamp

        # Update index
        segment_data['index'] = range(len(segment_data))

        # Save parquet file
        parquet_path = self.output_path / "data" / "chunk-000" / f"episode_{new_episode_id:06d}.parquet"
        segment_data.to_parquet(parquet_path, index=False)

        # Export video segments for all cameras
        for camera_name, video_processor in video_processor_map.items():
            video_output_path = (
                self.output_path / "videos" / "chunk-000" /
                f"observation.images.{camera_name}" / f"episode_{new_episode_id:06d}.mp4"
            )

            if camera_name == "main" and perspective_corrector.is_calibrated():
                # Apply perspective correction to main camera
                def transform_fn(frame):
                    return perspective_corrector.correct_image(frame, perspective_corrector.output_size)

                video_processor.export_segment(
                    segment.start_frame,
                    segment.end_frame,
                    str(video_output_path),
                    transform_fn=transform_fn,
                    output_size=perspective_corrector.output_size
                )
            else:
                # For wrist cameras, just extract segment without transformation
                video_processor.export_segment(
                    segment.start_frame,
                    segment.end_frame,
                    str(video_output_path)
                )

        # Create episode metadata
        episode_metadata = {
            "episode_index": new_episode_id,
            "tasks": [segment.primitive.primitive_type.value],
            "length": len(segment_data),
            "primitive_annotation": {
                "type": segment.primitive.primitive_type.value,
                "coordinates": [c.to_list() for c in segment.primitive.coordinates],
                "target_position": segment.primitive.target_position.to_list() if segment.primitive.target_position else None,
                "primitive_string": segment.primitive.to_string()
            },
            "source_episode": segment.episode_id,
            "source_frame_start": segment.start_frame,
            "source_frame_end": segment.end_frame
        }

        return episode_metadata

    def compute_and_save_stats(self, episode_data_list: List[pd.DataFrame]):
        """
        Compute and save dataset statistics.

        Args:
            episode_data_list: List of all episode dataframes
        """
        # Stack all action and state data
        all_actions = []
        all_states = []

        for episode_data in episode_data_list:
            if 'action' in episode_data.columns:
                actions = np.stack(episode_data['action'].values)
                all_actions.append(actions)

            if 'observation.state' in episode_data.columns:
                states = np.stack(episode_data['observation.state'].values)
                all_states.append(states)

        # Compute statistics
        stats = {}

        if all_actions:
            all_actions = np.vstack(all_actions)
            stats['action'] = {
                'mean': all_actions.mean(axis=0).tolist(),
                'std': all_actions.std(axis=0).tolist(),
                'min': all_actions.min(axis=0).tolist(),
                'max': all_actions.max(axis=0).tolist()
            }

        if all_states:
            all_states = np.vstack(all_states)
            stats['observation.state'] = {
                'mean': all_states.mean(axis=0).tolist(),
                'std': all_states.std(axis=0).tolist(),
                'min': all_states.min(axis=0).tolist(),
                'max': all_states.max(axis=0).tolist()
            }

        # Save to episodes_stats.jsonl (one line per stat)
        stats_path = self.output_path / "meta" / "episodes_stats.jsonl"
        with open(stats_path, 'w') as f:
            for key, value in stats.items():
                f.write(json.dumps({key: value}) + '\n')

    def export_all_segments(self, segments: List[TrajectorySegment],
                          episode_data_map: Dict[int, pd.DataFrame],
                          video_processor_map: Dict[int, Dict[str, VideoProcessor]],
                          perspective_corrector: PerspectiveCorrector,
                          progress_callback=None) -> List[Dict]:
        """
        Export all segments as new dataset.

        Args:
            segments: All annotated segments
            episode_data_map: {episode_id: DataFrame}
            video_processor_map: {episode_id: {camera_name: VideoProcessor}}
            perspective_corrector: Perspective corrector for main camera
            progress_callback: Optional callback for progress updates

        Returns:
            List of episode metadata dicts
        """
        self.create_dataset_structure()

        episode_metadata_list = []
        episode_data_list = []

        for idx, segment in enumerate(segments):
            if progress_callback:
                progress_callback(idx, len(segments), f"Exporting segment {idx + 1}/{len(segments)}")

            episode_data = episode_data_map[segment.episode_id]
            video_processors = video_processor_map[segment.episode_id]

            episode_metadata = self.export_segment(
                segment,
                episode_data,
                video_processors,
                perspective_corrector,
                idx
            )

            episode_metadata_list.append(episode_metadata)

            # Load exported parquet for statistics computation
            parquet_path = self.output_path / "data" / "chunk-000" / f"episode_{idx:06d}.parquet"
            episode_data_list.append(pd.read_parquet(parquet_path))

        # Compute and save statistics
        if progress_callback:
            progress_callback(len(segments), len(segments), "Computing dataset statistics...")

        self.compute_and_save_stats(episode_data_list)

        # Create metadata files
        self.create_metadata(episode_metadata_list)

        if progress_callback:
            progress_callback(len(segments), len(segments), "Export complete!")

        return episode_metadata_list

    def create_metadata(self, episode_metadata_list: List[Dict]):
        """
        Generate meta/info.json, meta/episodes.jsonl, meta/tasks.jsonl.

        Args:
            episode_metadata_list: List of episode metadata dicts
        """
        # Create info.json
        info = self.original_metadata.copy()
        info["total_episodes"] = len(episode_metadata_list)
        info["total_frames"] = sum(ep["length"] for ep in episode_metadata_list)
        info["splits"] = {"train": f"0:{len(episode_metadata_list)}"}

        # Get unique tasks
        all_tasks = set()
        for ep in episode_metadata_list:
            all_tasks.update(ep["tasks"])
        info["total_tasks"] = len(all_tasks)

        # Count total videos
        camera_count = sum(1 for key in info["features"] if key.startswith("observation.images."))
        info["total_videos"] = len(episode_metadata_list) * camera_count

        # Update image dimensions if perspective correction was applied
        # (This assumes all corrected images have the same size)

        with open(self.output_path / "meta" / "info.json", 'w') as f:
            json.dump(info, f, indent=2)

        # Create episodes.jsonl
        with open(self.output_path / "meta" / "episodes.jsonl", 'w') as f:
            for ep_meta in episode_metadata_list:
                f.write(json.dumps(ep_meta) + '\n')

        # Create tasks.jsonl
        tasks = [{"task_index": i, "task": task} for i, task in enumerate(sorted(all_tasks))]
        with open(self.output_path / "meta" / "tasks.jsonl", 'w') as f:
            for task in tasks:
                f.write(json.dumps(task) + '\n')
