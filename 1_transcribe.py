"""
DEPRECATED: This file is kept for backward compatibility only.
Use 'transcriber.py' instead. This file will be removed in a future version.
"""
import warnings
warnings.warn(
    "1_transcribe.py is deprecated. Use 'transcriber.py' instead.",
    DeprecationWarning, stacklevel=1,
)

# Re-export from the new module for backward compatibility
from transcriber import transcribe_video, format_timestamp_ass, create_word_chunks, save_ass_hormozi

if __name__ == "__main__":
    import sys, os
    VIDEO_PATH = sys.argv[1] if len(sys.argv) > 1 else "input_video.mp4"
    if not os.path.exists(VIDEO_PATH):
        print(f"Error: {VIDEO_PATH} not found.")
    else:
        transcribe_video(VIDEO_PATH, "transcript.json", "subtitles.ass")
