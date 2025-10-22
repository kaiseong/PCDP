"""
Concatenates all 'orbbec_video.mp4' files from episode directories
into a single video file, adding a text overlay indicating the episode number.

This script uses ffmpeg and requires it to be installed and in the system's PATH.
The process involves re-encoding, so it will be slower than simple concatenation.
"""

import os
import glob
import subprocess
from pathlib import Path
import shutil

# ==================== USER CONFIGURATION ====================
# Set the base directory where your 'videos' folder is located.
BASE_DIR = "/home/nscl/diffusion_policy/"

# Set the desired name for the final output file.
OUTPUT_FILENAME = "concatenated_orbbec_videos_with_text.mp4"
# ============================================================

def find_and_sort_videos(videos_dir: Path) -> list[str]:
    """Finds all orbbec_video.mp4 files and sorts them numerically by episode."""
    # Correct pattern for numbered directories (e.g., '0000', '0001')
    search_pattern = str(videos_dir / "*" / "orbbec_video.mp4")
    found_videos = glob.glob(search_pattern)

    if not found_videos:
        return []

    def get_episode_number(path_str: str) -> int:
        """Extracts the episode number from a path like '.../videos/0001/...'."""
        try:
            episode_folder_name = Path(path_str).parent.name
            return int(episode_folder_name)
        except (ValueError, IndexError):
            return float('inf')

    return sorted(found_videos, key=get_episode_number)

def add_text_overlays(video_files: list[str], temp_dir: Path) -> list[str]:
    """
    Adds episode number text overlay to each video, saving them as temporary files.
    Returns a list of paths to the new temporary video files.
    """
    temp_video_paths = []
    print(f"\nStarting to add text overlays to {len(video_files)} videos...")
    print("This will take some time as it requires re-encoding.")

    for i, video_file in enumerate(video_files):
        episode_number = Path(video_file).parent.name
        output_temp_file = temp_dir / f"temp_{i:04d}.mp4"
        temp_video_paths.append(str(output_temp_file))

        # Escape the colon for the ffmpeg drawtext filter
        text = f"Episode\\:{episode_number}"

        command = [
            'ffmpeg',
            '-y',
            '-i', video_file,
            # -vf is for video filter. The drawtext filter adds the text.
            '-vf', f"drawtext=text='{text}':x=10:y=10:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.5",
            # Video codec settings for H.264
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '23',
            # Copy audio stream without re-encoding
            '-c:a', 'copy',
            str(output_temp_file)
        ]

        print(f"\nProcessing [{i+1}/{len(video_files)}]: {video_file}")
        try:
            subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] ffmpeg failed while adding text to {video_file}.")
            print(f"ffmpeg stderr:\n{e.stderr}")
            # If one video fails, stop the whole process
            return []

    print("\nFinished adding text overlays.")
    return temp_video_paths

def create_ffmpeg_file_list(video_files: list[str], list_filename: str):
    """Creates a temporary text file for ffmpeg's concat demuxer."""
    with open(list_filename, 'w') as f:
        for video_file in video_files:
            safe_path = Path(video_file).as_posix()
            f.write(f"file '{safe_path}'\n")

def run_ffmpeg_concat(file_list: str, output_file: str):
    """
    Executes the ffmpeg command to concatenate videos without re-encoding.
    """
    command = [
        'ffmpeg',
        '-y',
        '-f', 'concat',
        '-safe', '0',
        '-i', file_list,
        '-c', 'copy',
        output_file
    ]

    print("\nExecuting final concatenation command:")
    print(f"{' '.join(command)}")
    try:
        subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
        print(f"\nSuccessfully created final concatenated video: {output_file}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Final concatenation failed.")
        print(f"ffmpeg stderr:\n{e.stderr}")
        return False

if __name__ == "__main__":
    base_path = Path(BASE_DIR)
    videos_directory = base_path / "data/please_please/videos"
    output_path = base_path / OUTPUT_FILENAME
    temp_dir = base_path / "temp_videos_for_concat"
    temp_file_list = temp_dir / "temp_video_files.txt"

    # Create a temporary directory for intermediate files
    if not temp_dir.exists():
        temp_dir.mkdir()

    try:
        print(f"Searching for videos in: {videos_directory}")
        source_videos = find_and_sort_videos(videos_directory)

        if not source_videos:
            print("No 'orbbec_video.mp4' files found to concatenate.")
        else:
            print(f"Found {len(source_videos)} video files.")
            
            # Step 1: Add text overlay to each video and save as temp files
            temp_videos = add_text_overlays(source_videos, temp_dir)

            if temp_videos:
                # Step 2: Create a file list of the new temp videos
                create_ffmpeg_file_list(temp_videos, str(temp_file_list))
                
                # Step 3: Concatenate the temp videos
                success = run_ffmpeg_concat(str(temp_file_list), str(output_path))
                if success:
                    print("\nDone.")
                else:
                    print("\nProcess finished with errors.")

    finally:
        # Clean up the temporary directory and its contents
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            print(f"\nCleaned up temporary directory: {temp_dir}")