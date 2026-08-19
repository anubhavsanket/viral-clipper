"""
DEPRECATED: This file is kept for backward compatibility only.
Use 'report_generator.py' instead. This file will be removed in a future version.
"""
import warnings
warnings.warn(
    "4_generate_report.py is deprecated. Use 'report_generator.py' instead.",
    DeprecationWarning, stacklevel=1,
)

from report_generator import generate_report

if __name__ == "__main__":
    generate_report("clips.json", "transcript.json", "final_clips")
