import os
import subprocess
from pathlib import Path

def convert_av1_to_h264(root_dir):
    """
    Recursively finds all .mp4 files in root_dir and converts them from AV1 to H.264.
    The original files are replaced.
    """
    video_root = Path(root_dir)
    for video_path in video_root.rglob("*.mp4"):
        print(f"Processing {video_path}...")
        
        # Create a temporary output path
        temp_output_path = video_path.with_suffix(".temp.mp4")
        
        # ffmpeg command to convert video to h264
        # -c:v libx264 sets the video codec
        # -pix_fmt yuv420p is important for compatibility
        # -y overwrites the output file if it exists
        command = [
            "ffmpeg",
            "-i", str(video_path),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-y",
            str(temp_output_path)
        ]
        
        try:
            # Run the command
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            
            # If successful, replace the original file
            os.replace(temp_output_path, video_path)
            print(f"Successfully converted {video_path}")
            
        except subprocess.CalledProcessError as e:
            print(f"Failed to convert {video_path}.")
            print(f"ffmpeg stderr: {e.stderr}")
            # Clean up temp file on failure
            if temp_output_path.exists():
                os.remove(temp_output_path)
        except Exception as e:
            print(f"An unexpected error occurred with {video_path}: {e}")


if __name__ == "__main__":
    # IMPORTANT: Replace with the actual path to your dataset's video directory
    dataset_video_folder = "sweep2E_dualarm_v1/videos"
    
    if not Path(dataset_video_folder).is_dir():
        print(f"Error: Directory not found at '{dataset_video_folder}'")
        print("Please make sure the path is correct.")
    else:
        print(f"Starting video conversion in '{dataset_video_folder}'...")
        convert_av1_to_h264(dataset_video_folder)
        print("Conversion process finished.")
