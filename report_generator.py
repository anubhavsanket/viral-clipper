"""
Report generation for viral clip analysis.
Produces a Markdown report and an engagement timeline chart.
"""

import json
import os
from typing import Any, Dict, List, Optional


def generate_report(
    clips_json: str,
    transcript_json: str,
    output_dir: str,
) -> None:
    """Generate a virality report with clip summary and engagement timeline chart.

    Args:
        clips_json: Path to clips JSON file.
        transcript_json: Path to transcript JSON file.
        output_dir: Directory to write report files.

    Raises:
        FileNotFoundError: If input files don't exist.
        ValueError: If data is malformed.
    """
    report_file = os.path.join(output_dir, "VIRALITY_REPORT.md")
    chart_file = os.path.join(output_dir, "engagement_chart.png")

    if not os.path.exists(clips_json) or not os.path.exists(transcript_json):
        raise FileNotFoundError("JSON files not found. Run analysis first.")

    # Load data with validation
    with open(clips_json, "r", encoding="utf-8") as f:
        clips: List[Dict[str, Any]] = json.load(f)

    with open(transcript_json, "r", encoding="utf-8") as f:
        transcript: List[Dict[str, Any]] = json.load(f)

    # Safely extract video duration
    video_duration = 0.0
    if transcript:
        for seg in reversed(transcript):
            if isinstance(seg, dict) and "end" in seg:
                try:
                    video_duration = float(seg["end"])
                    break
                except (ValueError, TypeError):
                    continue

    # --- 1. Generate Markdown Report ---
    md_lines: List[str] = []
    md_lines.append("## AI Virality Report\n")
    md_lines.append(f"**Total Video Duration:** {video_duration / 60:.2f} minutes\n")
    md_lines.append(f"**Clips Generated:** {len(clips)}\n\n")

    md_lines.append("## Viral Clips Summary\n")
    md_lines.append("| Clip # | Time Range | Duration | Score | Reasoning |\n")
    md_lines.append("| :--- | :--- | :--- | :--- | :--- |\n")

    for i, clip in enumerate(clips):
        start_time = clip.get("start_time", 0)
        end_time = clip.get("end_time", 0)
        duration = clip.get("duration", 0)

        start = f"{int(start_time // 60)}:{int(start_time % 60):02d}"
        end = f"{int(end_time // 60)}:{int(end_time % 60):02d}"
        score = clip.get("virality_score", "N/A")
        reason = clip.get("reasoning", "No reasoning provided.").replace("\n", " ")

        md_lines.append(f"| {i + 1} | {start} - {end} | {duration}s | **{score}/100** | {reason} |\n")

    md_lines.append("\n## Visual Timeline\n")
    md_lines.append("![Engagement Chart](./engagement_chart.png)\n")

    md_content = "".join(md_lines)

    # --- 2. Generate Chart ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))

    # Gray background bar for full video
    ax.barh(y=1, width=video_duration, left=0, height=0.5, color="#e0e0e0", label="Full Video")

    # Colored bars for clips
    colors = ["#FF4B4B", "#FF8F4B", "#FFD44B", "#4BFF8F", "#4BAAFF"]
    for i, clip in enumerate(clips):
        c = colors[i % len(colors)]
        clip_duration = clip.get("duration", 0)
        clip_start = clip.get("start_time", 0)
        score = clip.get("virality_score", "?")

        ax.barh(
            y=1, width=clip_duration, left=clip_start,
            height=0.5, color=c, edgecolor="black",
            label=f"Clip {i + 1}",
        )
        ax.text(
            clip_start, 1.3,
            f"Clip {i + 1} ({score})",
            fontsize=9, rotation=45,
        )

    ax.set_yticks([])
    ax.set_xlabel("Time (seconds)")
    ax.set_title("Viral Segments Timeline")
    ax.set_xlim(0, video_duration + 10)
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(chart_file, dpi=150, bbox_inches="tight")
    plt.close(fig)  # Prevent memory leak
    print(f"--- Chart saved to {chart_file} ---")

    # --- 3. Save Markdown ---
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"--- Report saved to {report_file} ---")


if __name__ == "__main__":
    generate_report("clips.json", "transcript.json", "final_clips")
