"""
DEPRECATED: This file is kept for backward compatibility only.
Use 'analyzer.py' instead. This file will be removed in a future version.
"""
import warnings
warnings.warn(
    "2_analyze.py is deprecated. Use 'analyzer.py' instead.",
    DeprecationWarning, stacklevel=1,
)

from analyzer import analyze_transcript

if __name__ == "__main__":
    analyze_transcript("transcript.json", "clips.json")
