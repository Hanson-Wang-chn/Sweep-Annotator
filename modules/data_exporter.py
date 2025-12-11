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
import multiprocessing as mp


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

        DEPRECATED: This method is no longer used. Use export_all_segments() instead,
        which properly handles task_index mapping and parallel processing.

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

        # Create episode metadata (LeRobot 2.1 format - only episode_index, tasks, length)
        # tasks field now contains full primitive string
        episode_metadata = {
            "episode_index": new_episode_id,
            "tasks": [segment.primitive.to_string()],
            "length": len(segment_data)
        }

        return episode_metadata

    def compute_and_save_stats(self, episode_data_list: List[pd.DataFrame]):
        """
        Compute and save per-episode statistics in LeRobot 2.1 format.

        Args:
            episode_data_list: List of all episode dataframes
        """
        stats_path = self.output_path / "meta" / "episodes_stats.jsonl"

        with open(stats_path, 'w') as f:
            for episode_index, episode_data in enumerate(episode_data_list):
                # Compute statistics for this episode
                episode_stats = {
                    "episode_index": episode_index,
                    "stats": {}
                }

                # Add action statistics
                if 'action' in episode_data.columns:
                    actions = np.stack(episode_data['action'].values)
                    episode_stats['stats']['action'] = {
                        'min': actions.min(axis=0).tolist(),
                        'max': actions.max(axis=0).tolist(),
                        'mean': actions.mean(axis=0).tolist(),
                        'std': actions.std(axis=0).tolist(),
                        'count': [len(actions)]
                    }

                # Add observation.state statistics
                if 'observation.state' in episode_data.columns:
                    states = np.stack(episode_data['observation.state'].values)
                    episode_stats['stats']['observation.state'] = {
                        'min': states.min(axis=0).tolist(),
                        'max': states.max(axis=0).tolist(),
                        'mean': states.mean(axis=0).tolist(),
                        'std': states.std(axis=0).tolist(),
                        'count': [len(states)]
                    }

                # Add image statistics for each camera
                for feature_name in self.original_metadata.get("features", {}):
                    if feature_name.startswith("observation.images."):
                        # Count video frames (typically sampled at lower fps than control)
                        video_count = len(episode_data)  # This is an approximation
                        # For proper video stats, we'd need to load actual video frames
                        # For now, we'll add placeholder stats matching the original format
                        episode_stats['stats'][feature_name] = {
                            'min': [[[0.0]], [[0.0]], [[0.0]]],
                            'max': [[[1.0]], [[1.0]], [[1.0]]],
                            'mean': [[[0.5]], [[0.5]], [[0.5]]],
                            'std': [[[0.25]], [[0.25]], [[0.25]]],
                            'count': [video_count]
                        }

                # Add timestamp statistics
                if 'timestamp' in episode_data.columns:
                    timestamps = episode_data['timestamp'].values
                    episode_stats['stats']['timestamp'] = {
                        'min': [float(timestamps.min())],
                        'max': [float(timestamps.max())],
                        'mean': [float(timestamps.mean())],
                        'std': [float(timestamps.std())],
                        'count': [len(timestamps)]
                    }

                # Add frame_index statistics
                if 'frame_index' in episode_data.columns:
                    frame_indices = episode_data['frame_index'].values
                    episode_stats['stats']['frame_index'] = {
                        'min': [int(frame_indices.min())],
                        'max': [int(frame_indices.max())],
                        'mean': [float(frame_indices.mean())],
                        'std': [float(frame_indices.std())],
                        'count': [len(frame_indices)]
                    }

                # Add episode_index statistics
                episode_stats['stats']['episode_index'] = {
                    'min': [episode_index],
                    'max': [episode_index],
                    'mean': [float(episode_index)],
                    'std': [0.0],
                    'count': [len(episode_data)]
                }

                # Add index statistics
                if 'index' in episode_data.columns:
                    indices = episode_data['index'].values
                    episode_stats['stats']['index'] = {
                        'min': [int(indices.min())],
                        'max': [int(indices.max())],
                        'mean': [float(indices.mean())],
                        'std': [float(indices.std())],
                        'count': [len(indices)]
                    }

                # Add task_index statistics
                if 'task_index' in episode_data.columns:
                    task_indices = episode_data['task_index'].values
                    episode_stats['stats']['task_index'] = {
                        'min': [int(task_indices.min())],
                        'max': [int(task_indices.max())],
                        'mean': [float(task_indices.mean())],
                        'std': [float(task_indices.std())],
                        'count': [len(task_indices)]
                    }

                # Write this episode's stats as one line
                f.write(json.dumps(episode_stats) + '\n')

    def export_all_segments(self, segments: List[TrajectorySegment],
                          episode_data_map: Dict[int, pd.DataFrame],
                          video_processor_map: Dict[int, Dict[str, VideoProcessor]],
                          perspective_corrector: PerspectiveCorrector,
                          progress_callback=None,
                          num_workers: int = 8) -> List[Dict]:
        """
        Export all segments as new dataset using multiprocessing.

        Args:
            segments: All annotated segments
            episode_data_map: {episode_id: DataFrame}
            video_processor_map: {episode_id: {camera_name: VideoProcessor}}
            perspective_corrector: Perspective corrector for main camera
            progress_callback: Optional callback for progress updates
            num_workers: Number of parallel processes to use (default: 8)

        Returns:
            List of episode metadata dicts
        """
        self.create_dataset_structure()

        # PHASE 1: Collect all unique tasks and create task_index mapping
        if progress_callback:
            progress_callback(0, len(segments), "Building task index mapping...")

        all_tasks = set()
        for segment in segments:
            all_tasks.add(segment.primitive.to_string())

        # Create task to task_index mapping (sorted for consistency)
        task_to_index = {task: idx for idx, task in enumerate(sorted(all_tasks))}

        if progress_callback:
            progress_callback(0, len(segments), f"Found {len(all_tasks)} unique tasks. Starting parallel export...")

        episode_metadata_list = []
        episode_data_list = []

        # PHASE 2: Prepare arguments for parallel processing with task_index mapping
        export_args = []
        for idx, segment in enumerate(segments):
            episode_data = episode_data_map[segment.episode_id]
            video_processors = video_processor_map[segment.episode_id]

            # Get the task_index for this segment's task
            task_string = segment.primitive.to_string()
            task_index = task_to_index[task_string]

            # Serialize the arguments
            export_args.append({
                'segment': segment,
                'episode_data': episode_data,
                'video_paths': {cam: vp.video_path for cam, vp in video_processors.items()},
                'perspective_corrector_data': {
                    'is_calibrated': perspective_corrector.is_calibrated(),
                    'src_points': perspective_corrector.src_points.tolist() if perspective_corrector.src_points is not None else None,
                    'dst_points': perspective_corrector.dst_points.tolist() if perspective_corrector.dst_points is not None else None,
                    'output_size': perspective_corrector.output_size
                },
                'new_episode_id': idx,
                'task_index': task_index,  # Pass the correct task_index
                'output_path': str(self.output_path),
                'original_metadata': self.original_metadata
            })

        # Use ProcessPoolExecutor for parallel processing
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            future_to_idx = {
                executor.submit(_export_segment_worker, args): args['new_episode_id']
                for args in export_args
            }

            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    episode_metadata = future.result()
                    episode_metadata_list.append((idx, episode_metadata))

                    completed += 1
                    if progress_callback:
                        progress_callback(completed, len(segments),
                                        f"Exported segment {completed}/{len(segments)}")
                except Exception as e:
                    print(f"Error exporting segment {idx}: {str(e)}")
                    raise

        # Sort by episode index to maintain order
        episode_metadata_list.sort(key=lambda x: x[0])
        episode_metadata_list = [meta for _, meta in episode_metadata_list]

        # Load exported parquet files for statistics computation
        if progress_callback:
            progress_callback(len(segments), len(segments), "Loading exported data for statistics...")

        for idx in range(len(segments)):
            parquet_path = self.output_path / "data" / "chunk-000" / f"episode_{idx:06d}.parquet"
            episode_data_list.append(pd.read_parquet(parquet_path))

        # Compute and save statistics
        if progress_callback:
            progress_callback(len(segments), len(segments), "Computing dataset statistics...")

        self.compute_and_save_stats(episode_data_list)

        # Create metadata files (now with pre-computed task mapping)
        self.create_metadata(episode_metadata_list, task_to_index)

        if progress_callback:
            progress_callback(len(segments), len(segments), "Export complete!")

        return episode_metadata_list

    def create_metadata(self, episode_metadata_list: List[Dict], task_to_index: Dict[str, int]):
        """
        Generate meta/info.json, meta/episodes.jsonl, meta/tasks.jsonl.

        Args:
            episode_metadata_list: List of episode metadata dicts
            task_to_index: Mapping from task string to task_index
        """
        # Create info.json
        info = self.original_metadata.copy()
        info["total_episodes"] = len(episode_metadata_list)
        info["total_frames"] = sum(ep["length"] for ep in episode_metadata_list)
        info["splits"] = {"train": f"0:{len(episode_metadata_list)}"}

        # Use pre-computed task mapping
        info["total_tasks"] = len(task_to_index)

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

        # Create tasks.jsonl using pre-computed task mapping
        tasks = [{"task_index": idx, "task": task} for task, idx in sorted(task_to_index.items(), key=lambda x: x[1])]
        with open(self.output_path / "meta" / "tasks.jsonl", 'w') as f:
            for task in tasks:
                f.write(json.dumps(task) + '\n')


def _export_segment_worker(args: Dict) -> Dict:
    """
    Worker function for parallel segment export.
    This function runs in a separate process.

    Args:
        args: Dictionary containing all necessary arguments

    Returns:
        Episode metadata dict
    """
    import numpy as np
    from pathlib import Path
    from modules.video_processor import VideoProcessor
    from modules.perspective_corrector import PerspectiveCorrector

    # Unpack arguments
    segment = args['segment']
    episode_data = args['episode_data']
    video_paths = args['video_paths']
    perspective_data = args['perspective_corrector_data']
    new_episode_id = args['new_episode_id']
    task_index = args['task_index']  # Get the correct task_index
    output_path = Path(args['output_path'])
    original_metadata = args['original_metadata']

    # Reconstruct perspective corrector
    perspective_corrector = PerspectiveCorrector(output_size=tuple(perspective_data['output_size']))
    if perspective_data['is_calibrated']:
        src_points = np.array(perspective_data['src_points'], dtype=np.float32)
        dst_points = np.array(perspective_data['dst_points'], dtype=np.float32)
        perspective_corrector.set_correction_points(src_points, dst_points)

    # Create video processors
    video_processor_map = {}
    for camera_name, video_path in video_paths.items():
        video_processor_map[camera_name] = VideoProcessor(video_path)

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

    # Set the correct task_index for all rows in this segment
    segment_data['task_index'] = task_index

    # Save parquet file
    parquet_path = output_path / "data" / "chunk-000" / f"episode_{new_episode_id:06d}.parquet"
    segment_data.to_parquet(parquet_path, index=False)

    # Export video segments for all cameras
    for camera_name, video_processor in video_processor_map.items():
        video_output_path = (
            output_path / "videos" / "chunk-000" /
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

    # Clean up video processors
    for vp in video_processor_map.values():
        vp.release()

    # Create episode metadata
    episode_metadata = {
        "episode_index": new_episode_id,
        "tasks": [segment.primitive.to_string()],
        "length": len(segment_data)
    }

    return episode_metadata
