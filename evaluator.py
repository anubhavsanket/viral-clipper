"""
Automated clip evaluation pipeline.
Scores output clips on audio quality, visual quality, caption accuracy, and engagement potential.
"""

import json
import os
import subprocess
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ClipScore:
    """Scores for a single clip across multiple dimensions."""
    overall: float = 0.0
    audio_score: float = 0.0
    visual_score: float = 0.0
    caption_score: float = 0.0
    engagement_score: float = 0.0
    duration_score: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


def _get_ffprobe_path() -> str:
    for candidate in ["ffprobe", "bin/ffprobe.exe", "ffprobe.exe"]:
        if os.path.exists(candidate):
            return candidate
    return "ffprobe"


def analyze_audio(clip_path: str) -> Dict[str, float]:
    """Analyze audio quality of a clip.

    Returns:
        Dict with loudness_db, peak_db, dynamic_range, speech_clarity.
    """
    ffprobe = _get_ffprobe_path()

    # Get loudness stats
    cmd = [
        "ffmpeg", "-i", clip_path,
        "-af", "ebur128=peak=true",
        "-f", "null", "-",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        stderr = result.stderr

        # Parse integrated loudness
        loudness = -23.0  # default
        peak = -1.0
        for line in stderr.split("\n"):
            if "I:" in line and "LUFS" in line:
                try:
                    loudness = float(line.split("I:")[1].split("LUFS")[0].strip())
                except (ValueError, IndexError):
                    pass
            if "Peak:" in line and "dBTP" in line:
                try:
                    peak = float(line.split("Peak:")[1].split("dBTP")[0].strip())
                except (ValueError, IndexError):
                    pass

        # Score: ideal loudness is -14 to -16 LUFS for social media
        loudness_score = max(0, 100 - abs(loudness + 15) * 5)
        # Score: peak should be close to 0 but not clipped
        peak_score = max(0, 100 - abs(peak + 1) * 10) if peak < 0 else max(0, 100 - (peak * 50))
        # Dynamic range: 6-12 dB is good for speech
        dynamic_range = abs(loudness - peak)
        dr_score = max(0, 100 - abs(dynamic_range - 9) * 8)

        return {
            "loudness_db": round(loudness, 1),
            "peak_db": round(peak, 1),
            "dynamic_range": round(dynamic_range, 1),
            "loudness_score": round(min(100, max(0, loudness_score)), 1),
            "peak_score": round(min(100, max(0, peak_score)), 1),
            "dr_score": round(min(100, max(0, dr_score)), 1),
        }
    except Exception:
        return {"loudness_score": 50, "peak_score": 50, "dr_score": 50}


def analyze_visual(clip_path: str) -> Dict[str, float]:
    """Analyze visual quality of a clip.

    Returns:
        Dict with brightness, contrast, blur_score.
    """
    try:
        import cv2
        import numpy as np

        cap = cv2.VideoCapture(clip_path)
        if not cap.isOpened():
            return {"brightness_score": 50, "contrast_score": 50, "blur_score": 50}

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_frames = min(10, frame_count)
        interval = max(1, frame_count // sample_frames)

        brightness_vals = []
        contrast_vals = []
        blur_vals = []

        for i in range(0, frame_count, interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Brightness: mean pixel value (ideal: 100-140)
            brightness = np.mean(gray)
            brightness_vals.append(brightness)

            # Contrast: standard deviation (ideal: 40-80)
            contrast = np.std(gray)
            contrast_vals.append(contrast)

            # Blur: Laplacian variance (higher = sharper)
            lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            blur_vals.append(lap_var)

        cap.release()

        avg_brightness = sum(brightness_vals) / max(len(brightness_vals), 1)
        avg_contrast = sum(contrast_vals) / max(len(contrast_vals), 1)
        avg_blur = sum(blur_vals) / max(len(blur_vals), 1)

        # Score brightness (ideal range 80-150)
        brightness_score = max(0, 100 - abs(avg_brightness - 115) * 1.5)
        # Score contrast (ideal range 35-75)
        contrast_score = max(0, 100 - abs(avg_contrast - 55) * 2)
        # Score sharpness (higher is better, >100 is good)
        blur_score = min(100, avg_blur / 2)

        return {
            "brightness": round(avg_brightness, 1),
            "contrast": round(avg_contrast, 1),
            "sharpness": round(avg_blur, 1),
            "brightness_score": round(min(100, max(0, brightness_score)), 1),
            "contrast_score": round(min(100, max(0, contrast_score)), 1),
            "blur_score": round(min(100, max(0, blur_score)), 1),
        }
    except ImportError:
        return {"brightness_score": 50, "contrast_score": 50, "blur_score": 50}
    except Exception:
        return {"brightness_score": 50, "contrast_score": 50, "blur_score": 50}


def analyze_captions(ass_path: str, clip_start: float, clip_end: float) -> Dict[str, float]:
    """Analyze subtitle quality.

    Returns:
        Dict with caption_count, words_per_subtitle, timing_coverage.
    """
    if not os.path.exists(ass_path):
        return {"caption_score": 50}

    try:
        with open(ass_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Parse dialogue lines
        dialogue_lines = [
            line for line in content.split("\n")
            if line.startswith("Dialogue:")
        ]

        # Filter to lines within clip time range
        clip_duration = clip_end - clip_start
        clip_lines = []
        total_subtitle_time = 0

        for line in dialogue_lines:
            parts = line.split(",", 9)
            if len(parts) < 10:
                continue
            start_str = parts[1]
            end_str = parts[2]
            text = parts[9]

            # Parse ASS timestamps
            def parse_ts(ts):
                parts = ts.strip().split(":")
                return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])

            sub_start = parse_ts(start_str)
            sub_end = parse_ts(end_str)

            # Check overlap with clip
            if sub_end > clip_start and sub_start < clip_end:
                clip_lines.append({"text": text, "words": len(text.split())})
                total_subtitle_time += min(sub_end, clip_end) - max(sub_start, clip_start)

        if not clip_lines:
            return {"caption_score": 30, "caption_count": 0}

        # Metrics
        caption_count = len(clip_lines)
        avg_words = sum(c["words"] for c in clip_lines) / caption_count
        coverage = total_subtitle_time / clip_duration if clip_duration > 0 else 0

        # Score: ideal is 2-3 words per subtitle, good coverage
        words_score = max(0, 100 - abs(avg_words - 2.5) * 30)
        coverage_score = min(100, coverage * 120)  # 80% coverage is ideal
        count_score = min(100, caption_count * 15)  # More captions = better engagement

        overall_caption = (words_score * 0.3 + coverage_score * 0.4 + count_score * 0.3)

        return {
            "caption_count": caption_count,
            "avg_words_per_sub": round(avg_words, 1),
            "coverage_pct": round(coverage * 100, 1),
            "words_score": round(words_score, 1),
            "coverage_score": round(coverage_score, 1),
            "count_score": round(count_score, 1),
            "caption_score": round(min(100, max(0, overall_caption)), 1),
        }
    except Exception:
        return {"caption_score": 50}


def score_duration(duration: float, target: int = 90) -> float:
    """Score how well the clip duration matches the target.

    Returns:
        Score 0-100.
    """
    diff = abs(duration - target)
    # Within 10% of target = 100, linearly decreasing to 0 at 50% off
    return max(0, min(100, 100 - (diff / target) * 200))


def evaluate_clip(
    clip_path: str,
    ass_path: str,
    clip_data: Dict[str, Any],
    target_duration: int = 90,
) -> ClipScore:
    """Full evaluation of a single clip.

    Args:
        clip_path: Path to the rendered clip video.
        ass_path: Path to the ASS subtitle file.
        clip_data: Clip metadata (start_time, end_time, virality_score, etc.).
        target_duration: Target duration for scoring.

    Returns:
        ClipScore with all dimension scores.
    """
    clip_start = clip_data.get("start_time", 0)
    clip_end = clip_data.get("end_time", 0)
    duration = clip_end - clip_start

    # Analyze each dimension
    audio = analyze_audio(clip_path)
    visual = analyze_visual(clip_path)
    captions = analyze_captions(ass_path, clip_start, clip_end)
    dur_score = score_duration(duration, target_duration)

    # Weighted overall score
    audio_s = (audio.get("loudness_score", 50) + audio.get("peak_score", 50) + audio.get("dr_score", 50)) / 3
    visual_s = (visual.get("brightness_score", 50) + visual.get("contrast_score", 50) + visual.get("blur_score", 50)) / 3
    caption_s = captions.get("caption_score", 50)
    engagement_s = clip_data.get("virality_score", 50)

    overall = (
        audio_s * 0.20
        + visual_s * 0.20
        + caption_s * 0.25
        + engagement_s * 0.20
        + dur_score * 0.15
    )

    return ClipScore(
        overall=round(overall, 1),
        audio_score=round(audio_s, 1),
        visual_score=round(visual_s, 1),
        caption_score=round(caption_s, 1),
        engagement_score=round(engagement_s, 1),
        duration_score=round(dur_score, 1),
        details={
            "audio": audio,
            "visual": visual,
            "captions": captions,
            "duration": round(duration, 1),
        },
    )


def evaluate_all_clips(
    clips_dir: str,
    clips_json_path: str,
    ass_path: str,
    target_duration: int = 90,
) -> List[Dict[str, Any]]:
    """Evaluate all rendered clips in a directory.

    Args:
        clips_dir: Directory containing rendered clip videos.
        clips_json_path: Path to clips.json with metadata.
        ass_path: Path to the ASS subtitle file.
        target_duration: Target duration for scoring.

    Returns:
        List of evaluation results per clip.
    """
    if not os.path.exists(clips_json_path):
        return []

    with open(clips_json_path, "r", encoding="utf-8") as f:
        clips_data = json.load(f)

    results = []
    for i, clip_data in enumerate(clips_data):
        # Find the rendered clip file (try multiple naming conventions)
        clip_file = None
        for pattern in [
            os.path.join(clips_dir, f"clip_{i + 1}.mp4"),       # clip_1.mp4 (renderer default)
            os.path.join(clips_dir, f"clip_{i + 1:02d}.mp4"),   # clip_01.mp4
            os.path.join(clips_dir, f"clip_{i}.mp4"),            # clip_0.mp4
            os.path.join(clips_dir, f"clip_{i:02d}.mp4"),       # clip_00.mp4
        ]:
            if os.path.exists(pattern):
                clip_file = pattern
                break

        if not clip_file:
            results.append({
                "clip_index": i,
                "error": "Rendered clip file not found",
                "score": ClipScore(overall=0),
            })
            continue

        score = evaluate_clip(clip_file, ass_path, clip_data, target_duration)
        results.append({
            "clip_index": i,
            "start_time": clip_data.get("start_time"),
            "end_time": clip_data.get("end_time"),
            "score": score,
        })

    return results


def generate_evaluation_report(
    results: List[Dict[str, Any]],
    output_path: str,
) -> str:
    """Generate a markdown evaluation report.

    Args:
        results: Results from evaluate_all_clips.
        output_path: Path to write the report.

    Returns:
        Path to the written report.
    """
    lines = ["# Clip Evaluation Report\n"]

    if not results:
        lines.append("No clips to evaluate.\n")
    else:
        avg_overall = sum(r["score"].overall for r in results if "score" in r) / max(len(results), 1)
        lines.append(f"**Average Quality Score: {avg_overall:.1f}/100**\n")
        lines.append("| Clip | Duration | Audio | Visual | Captions | Engagement | Overall |")
        lines.append("|------|----------|-------|--------|----------|------------|---------|")

        for r in results:
            if "error" in r:
                lines.append(f"| {r['clip_index']+1} | - | - | - | - | - | ERROR: {r['error']} |")
                continue
            s = r["score"]
            lines.append(
                f"| {r['clip_index']+1} | {s.details.get('duration', '?')}s "
                f"| {s.audio_score:.0f} | {s.visual_score:.0f} "
                f"| {s.caption_score:.0f} | {s.engagement_score:.0f} "
                f"| **{s.overall:.0f}** |"
            )

        lines.append("")
        lines.append("### Scoring Weights")
        lines.append("- Audio quality: 20%")
        lines.append("- Visual quality: 20%")
        lines.append("- Caption quality: 25%")
        lines.append("- Engagement (virality): 20%")
        lines.append("- Duration match: 15%")

    report = "\n".join(lines)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    return output_path
