"""
Batch processing runner for headless folder-based video processing.
Processes all videos in an input folder through the full pipeline.
"""

import os
import shutil
import subprocess
import sys
import time
from typing import List

from config import BASE_DIR, HANDBRAKE_ENCODER, SUPPORTED_VIDEO_EXTENSIONS


INPUT_FOLDER = "input_videos"
PROCESSED_FOLDER = "processed_videos"
TEMP_INPUT_NAME = "input_video.mp4"


def find_tool(tool_name: str) -> str:
    """Search for a tool in standard locations."""
    search_paths = [
        os.getcwd(),
        os.path.join(os.getcwd(), "bin"),
        os.path.dirname(os.path.abspath(__file__)),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin"),
        BASE_DIR,
        os.path.join(BASE_DIR, "bin"),
    ]
    for path in search_paths:
        full_path = os.path.join(path, tool_name)
        if os.path.exists(full_path):
            if path not in os.environ["PATH"]:
                os.environ["PATH"] += os.pathsep + path
            return full_path
    return tool_name


def check_tools() -> bool:
    """Verify required tools are available."""
    hb_path = find_tool("HandBrakeCLI.exe")
    if not os.path.exists(hb_path):
        print("Error: HandBrakeCLI.exe not found!")
        print("   Place it in the 'bin' folder.")
        return False
    return True


def convert_video_handbrake(src: str, dest: str) -> None:
    """Convert video using HandBrakeCLI with auto-detected encoder."""
    hb_path = find_tool("HandBrakeCLI.exe")
    print(f"Converting {src} with HandBrake ({HANDBRAKE_ENCODER})...")

    cmd = [
        hb_path, "--input", src, "--output", dest,
        "--format", "av_mp4",
        "--encoder", HANDBRAKE_ENCODER,
        "--quality", "20", "--cfr",
        "--aencoder", "copy",
        "--audio-fallback", "aac", "--ab", "192",
    ]

    try:
        subprocess.run(cmd, check=True)
        print("Conversion successful.")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"HandBrake conversion failed: {e}") from e


def process_video(filename: str) -> None:
    """Process a single video through the full pipeline."""
    print(f"\n{'=' * 50}")
    print(f"  PROCESSING: {filename}")
    print(f"{'=' * 50}")

    start_time = time.time()
    src = os.path.join(INPUT_FOLDER, filename)

    if os.path.exists(TEMP_INPUT_NAME):
        os.remove(TEMP_INPUT_NAME)

    # Step 0: HandBrake conversion
    try:
        convert_video_handbrake(src, TEMP_INPUT_NAME)
    except Exception:
        print("Skipping file due to conversion error.")
        return

    video_name_no_ext = os.path.splitext(filename)[0]
    final_dest = os.path.join(PROCESSED_FOLDER, video_name_no_ext)
    if os.path.exists(final_dest):
        shutil.rmtree(final_dest)
    os.makedirs(final_dest, exist_ok=True)

    try:
        from transcriber import transcribe_video
        from analyzer import analyze_transcript
        from clip_processor import process_clips
        from report_generator import generate_report

        transcript_json = os.path.join(final_dest, "transcript.json")
        subtitles_ass = os.path.join(final_dest, "subtitles.ass")
        clips_json = os.path.join(final_dest, "clips.json")

        print("\n[1/4] Transcribing Audio...")
        transcribe_video(TEMP_INPUT_NAME, transcript_json, subtitles_ass)

        print("\n[2/4] Analyzing Viral Hooks...")
        analyze_transcript(transcript_json, clips_json)

        print("\n[3/4] Rendering Clips...")
        process_clips(TEMP_INPUT_NAME, clips_json, subtitles_ass, final_dest)

        print("\n[4/4] Generating Report...")
        generate_report(clips_json, transcript_json, final_dest)

        duration = int(time.time() - start_time)
        print(f"Done! Results: {final_dest} ({duration}s)")

    except Exception as e:
        print(f"Pipeline failed: {e}")
    finally:
        if os.path.exists(TEMP_INPUT_NAME):
            os.remove(TEMP_INPUT_NAME)


if __name__ == "__main__":
    if not check_tools():
        sys.exit(1)

    os.makedirs(INPUT_FOLDER, exist_ok=True)
    os.makedirs(PROCESSED_FOLDER, exist_ok=True)

    files = [
        f for f in os.listdir(INPUT_FOLDER)
        if f.lower().endswith(tuple(SUPPORTED_VIDEO_EXTENSIONS))
    ]

    if not files:
        print(f"No videos found in '{INPUT_FOLDER}'.")
    else:
        print(f"Found {len(files)} videos. Starting batch pipeline...")
        for f in files:
            process_video(f)
