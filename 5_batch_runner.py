"""
DEPRECATED: This file is kept for backward compatibility only.
Use 'batch_runner.py' instead. This file will be removed in a future version.
"""
import warnings
warnings.warn(
    "5_batch_runner.py is deprecated. Use 'batch_runner.py' instead.",
    DeprecationWarning, stacklevel=1,
)

from batch_runner import process_video, check_tools, INPUT_FOLDER, PROCESSED_FOLDER, SUPPORTED_VIDEO_EXTENSIONS

import os
import sys


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
