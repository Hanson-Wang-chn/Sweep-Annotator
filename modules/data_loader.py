"""LeRobot dataset loading module."""
from pathlib import Path
from typing import List, Dict, Optional
import json
import pandas as pd
from config.primitives_config import LeRobotEpisode


class LeRobotDataLoader:
    """Load and manage LeRobot format datasets."""

    def __init__(self, dataset_path: str):
        """
        Initialize data loader.

        Args:
            dataset_path: Root path to LeRobot dataset
        """
        self.dataset_path = Path(dataset_path)
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

        self.metadata = None
        self.episodes_metadata = []
        self.tasks_metadata = []

    def load_dataset_metadata(self) -> Dict:
        """
        Load meta/info.json and meta/episodes.jsonl.

        Returns:
            dict: Dataset metadata including FPS, camera names, action dimensions
        """
        # Load info.json
        info_path = self.dataset_path / "meta" / "info.json"
        if not info_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {info_path}")

        with open(info_path, 'r') as f:
            self.metadata = json.load(f)

        # Load episodes.jsonl
        episodes_path = self.dataset_path / "meta" / "episodes.jsonl"
        if episodes_path.exists():
            with open(episodes_path, 'r') as f:
                self.episodes_metadata = [json.loads(line) for line in f]

        # Load tasks.jsonl
        tasks_path = self.dataset_path / "meta" / "tasks.jsonl"
        if tasks_path.exists():
            with open(tasks_path, 'r') as f:
                self.tasks_metadata = [json.loads(line) for line in f]

        return self.metadata

    def get_episode_list(self) -> List[int]:
        """
        Get list of available episode IDs.

        Returns:
            List of episode IDs available in dataset
        """
        if not self.episodes_metadata:
            self.load_dataset_metadata()

        return [ep["episode_index"] for ep in self.episodes_metadata]

    def get_episode_frame_count(self, episode_id: int) -> int:
        """
        Get total number of frames in episode.

        Args:
            episode_id: Episode number

        Returns:
            Total number of frames in episode
        """
        if not self.episodes_metadata:
            self.load_dataset_metadata()

        for ep in self.episodes_metadata:
            if ep["episode_index"] == episode_id:
                return ep["length"]

        raise ValueError(f"Episode {episode_id} not found in metadata")

    def load_episode(self, episode_id: int) -> LeRobotEpisode:
        """
        Load single episode data (parquet + video paths).

        Args:
            episode_id: Episode number

        Returns:
            LeRobotEpisode object
        """
        if self.metadata is None:
            self.load_dataset_metadata()

        # Get episode metadata
        episode_meta = None
        for ep in self.episodes_metadata:
            if ep["episode_index"] == episode_id:
                episode_meta = ep
                break

        if episode_meta is None:
            raise ValueError(f"Episode {episode_id} not found")

        # Construct paths
        episode_chunk = episode_id // self.metadata.get("chunks_size", 1000)

        # Parquet path
        parquet_path = self.dataset_path / self.metadata["data_path"].format(
            episode_chunk=episode_chunk,
            episode_index=episode_id
        )

        # Video paths for all cameras
        video_paths = {}
        camera_names = ["main", "secondary_0", "secondary_1"]
        for camera_name in camera_names:
            video_key = f"observation.images.{camera_name}"
            if video_key in self.metadata["features"]:
                video_path = self.dataset_path / self.metadata["video_path"].format(
                    episode_chunk=episode_chunk,
                    video_key=video_key,
                    episode_index=episode_id
                )
                if video_path.exists():
                    video_paths[camera_name] = str(video_path)

        # Load parquet data
        data = None
        if parquet_path.exists():
            data = pd.read_parquet(parquet_path)

        # Create episode object
        episode = LeRobotEpisode(
            episode_id=episode_id,
            total_frames=episode_meta["length"],
            fps=self.metadata["fps"],
            parquet_path=str(parquet_path),
            video_paths=video_paths,
            data=data
        )

        return episode

    def get_camera_names(self) -> List[str]:
        """
        Get list of camera names in dataset.

        Returns:
            List of camera names (e.g., ['main', 'secondary_0', 'secondary_1'])
        """
        if self.metadata is None:
            self.load_dataset_metadata()

        camera_names = []
        for feature_name in self.metadata["features"]:
            if feature_name.startswith("observation.images."):
                camera_name = feature_name.replace("observation.images.", "")
                camera_names.append(camera_name)

        return camera_names

    def get_action_dim(self) -> int:
        """
        Get action dimension.

        Returns:
            Action dimension
        """
        if self.metadata is None:
            self.load_dataset_metadata()

        return self.metadata["features"]["action"]["shape"][0]

    def get_state_dim(self) -> int:
        """
        Get observation state dimension.

        Returns:
            State dimension
        """
        if self.metadata is None:
            self.load_dataset_metadata()

        return self.metadata["features"]["observation.state"]["shape"][0]
