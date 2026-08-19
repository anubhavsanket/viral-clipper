"""
Video clip rendering engine using FFmpeg.
Handles cropping, subtitle burn-in, and encoding for short-form video output.
"""

import json
import os
import subprocess
from typing import Any, Dict, List, Optional, Tuple

from config import ClipConfig, USE_GPU, VIDEO_ENCODER


def create_clip(
    start_time: float,
    end_time: float,
    index: int,
    input_video: str,
    subtitles_file: str,
    output_dir: str,
    config: Optional[ClipConfig] = None,
    use_gpu: Optional[bool] = None,
) -> str:
    """Create a single video clip with burned-in subtitles and aspect ratio crop.

    Uses sync-safe FFmpeg filtering: loads video from 0:00, burns subtitles first,
    then trims to the desired time range. This ensures subtitle timestamps always
    align correctly regardless of seek precision.

    Args:
        start_time: Clip start time in seconds.
        end_time: Clip end time in seconds.
        index: Clip index (0-based) for naming.
        input_video: Path to the source video.
        subtitles_file: Path to the .ass subtitle file.
        output_dir: Directory to write the output clip.
        config: Clip rendering configuration.
        use_gpu: Force GPU/CPU encoding. None = auto-detect.

    Returns:
        Path to the created clip file.

    Raises:
        RuntimeError: If both GPU and CPU encoding fail.
    """
    if config is None:
        config = ClipConfig()

    if use_gpu is None:
        use_gpu = USE_GPU

    output_filename = os.path.join(output_dir, f"clip_{index + 1}.mp4")
    print(f"\n--- Processing Clip {index + 1} (Sync-Safe): {start_time}s to {end_time}s ---")

    # Clean paths for FFmpeg filter (forward slashes, escape colons)
    sub_path = subtitles_file.replace("\\", "/").replace(":", "\\:")

    # Compute crop dimensions from aspect ratio
    ratio_parts = config.aspect_ratio.split(":")
    try:
        ratio_w, ratio_h = int(ratio_parts[0]), int(ratio_parts[1])
    except (ValueError, IndexError):
        ratio_w, ratio_h = 9, 16  # Default vertical

    filter_complex = (
        f"[0:v]ass='{sub_path}',"
        f"crop=w=ih*({ratio_w}/{ratio_h}):h=ih:x=(iw-ow)/2:y=0,"
        f"trim=start={start_time}:end={end_time},"
        f"setpts=PTS-STARTPTS[v];"
        f"[0:a]atrim=start={start_time}:end={end_time},"
        f"asetpts=PTS-STARTPTS[a]"
    )

    cmd_gpu = [
        "ffmpeg", "-y",
        "-i", input_video,
        "-filter_complex", filter_complex,
        "-map", "[v]", "-map", "[a]",
        "-c:v", "h264_nvenc",
        "-pix_fmt", "yuv420p",
        "-preset", "p6",
        "-b:v", config.gpu_bitrate,
        "-c:a", config.audio_codec,
        "-b:a", config.audio_bitrate,
        output_filename,
    ]

    cmd_cpu = [
        "ffmpeg", "-y",
        "-i", input_video,
        "-filter_complex", filter_complex,
        "-map", "[v]", "-map", "[a]",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "medium",
        "-crf", str(config.cpu_crf),
        "-c:a", config.audio_codec,
        "-b:a", config.audio_bitrate,
        output_filename,
    ]

    # Try GPU first, then fall back to CPU
    if use_gpu:
        try:
            subprocess.run(cmd_gpu, check=True, stderr=subprocess.PIPE)
            print(f"Clip {index + 1} created (GPU).")
            return output_filename
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            stderr_text = ""
            if hasattr(e, "stderr") and e.stderr:
                stderr_text = e.stderr.decode(errors="replace")
            print(f"GPU encoding failed: {stderr_text[:200]}")
            print("Falling back to CPU...")

    # CPU fallback
    try:
        subprocess.run(cmd_cpu, check=True, stderr=subprocess.PIPE)
        print(f"Clip {index + 1} created (CPU).")
        return output_filename
    except subprocess.CalledProcessError as e:
        stderr_text = e.stderr.decode(errors="replace") if e.stderr else "Unknown error"
        raise RuntimeError(f"Clip {index + 1} failed: {stderr_text}") from e


def process_clips(
    input_video: str,
    clips_json: str,
    subtitles_file: str,
    output_dir: str,
    config: Optional[ClipConfig] = None,
    progress_callback=None,
) -> List[str]:
    """Render all clips from a clips JSON file.

    Args:
        input_video: Path to the source video.
        clips_json: Path to clips JSON file with timestamps.
        subtitles_file: Path to the .ass subtitle file.
        output_dir: Directory to write output clips.
        config: Clip rendering configuration.
        progress_callback: Optional callback for per-clip progress (0-100).

    Returns:
        List of paths to created clip files.

    Raises:
        FileNotFoundError: If input files don't exist.
    """
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(clips_json):
        raise FileNotFoundError(f"Clips JSON not found: {clips_json}")
    if not os.path.exists(subtitles_file):
        raise FileNotFoundError(f"Subtitles file not found: {subtitles_file}")

    with open(clips_json, "r") as f:
        clips = json.load(f)

    if not clips:
        print("No clips to process.")
        return []

    # GPU detection happens ONCE before the loop
    use_gpu = USE_GPU
    if use_gpu:
        print("--- GPU detected, using hardware encoding ---")
    else:
        print("--- No GPU detected, using CPU encoding ---")

    created_files: List[str] = []
    total = len(clips)

    for i, clip in enumerate(clips):
        try:
            filepath = create_clip(
                start_time=float(clip["start_time"]),
                end_time=float(clip["end_time"]),
                index=i,
                input_video=input_video,
                subtitles_file=subtitles_file,
                output_dir=output_dir,
                config=config,
                use_gpu=use_gpu,
            )
            created_files.append(filepath)

            # Per-clip progress
            if progress_callback:
                progress_callback(int(100 * (i + 1) / total))

        except RuntimeError as e:
            print(f"Warning: {e}")
            continue

    print(f"\n--- All {len(created_files)}/{total} clips processed! ---")
    return created_files


if __name__ == "__main__":
    INPUT_VIDEO = "input_video.mp4"
    CLIPS_JSON = "clips.json"
    OUTPUT_DIR = "final_clips"
    SUBTITLES_FILE = "subtitles.ass"

    if os.path.exists(CLIPS_JSON):
        process_clips(INPUT_VIDEO, CLIPS_JSON, SUBTITLES_FILE, OUTPUT_DIR)
