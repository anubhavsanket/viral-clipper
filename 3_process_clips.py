"""
DEPRECATED: This file is kept for backward compatibility only.
Use 'clip_processor.py' instead. This file will be removed in a future version.
"""
import warnings
warnings.warn(
    "3_process_clips.py is deprecated. Use 'clip_processor.py' instead.",
    DeprecationWarning, stacklevel=1,
)

from clip_processor import process_clips, create_clip

if __name__ == "__main__":
    import os
    INPUT_VIDEO = "input_video.mp4"
    CLIPS_JSON = "clips.json"
    OUTPUT_DIR = "final_clips"
    SUBTITLES_FILE = "subtitles.ass"
    if os.path.exists(CLIPS_JSON):
        process_clips(INPUT_VIDEO, CLIPS_JSON, SUBTITLES_FILE, OUTPUT_DIR)

