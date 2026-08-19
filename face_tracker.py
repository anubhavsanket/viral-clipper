"""
Face-tracking crop using MediaPipe Face Detection.
Analyzes video frames to find face positions, then computes smooth crop coordinates
that follow the speaker instead of naive center-crop.
"""

import json
import os
import subprocess
from typing import Any, Dict, List, Optional, Tuple


def _get_ffprobe_path() -> str:
    """Find ffprobe binary."""
    for candidate in ["ffprobe", "bin/ffprobe.exe", "ffprobe.exe"]:
        if os.path.exists(candidate):
            return candidate
    return "ffprobe"


def _get_ffmpeg_path() -> str:
    """Find ffmpeg binary."""
    for candidate in ["ffmpeg", "bin/ffmpeg.exe", "ffmpeg.exe"]:
        if os.path.exists(candidate):
            return candidate
    return "ffmpeg"


def get_video_info(video_path: str) -> Dict[str, Any]:
    """Get video dimensions and frame count."""
    ffprobe = _get_ffprobe_path()
    cmd = [
        ffprobe, "-v", "quiet", "-print_format", "json",
        "-show_streams", "-show_format", video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    info = json.loads(result.stdout)

    for stream in info.get("streams", []):
        if stream.get("codec_type") == "video":
            return {
                "width": int(stream["width"]),
                "height": int(stream["height"]),
                "fps": eval(stream.get("r_frame_rate", "30/1")),
                "duration": float(info.get("format", {}).get("duration", 0)),
            }
    return {"width": 1920, "height": 1080, "fps": 30, "duration": 0}


def detect_faces_in_segment(
    video_path: str,
    start_time: float,
    end_time: float,
    sample_interval: float = 0.5,
    max_samples: int = 20,
) -> List[Dict[str, Any]]:
    """Detect faces in a video segment by sampling frames.

    Args:
        video_path: Path to video file.
        start_time: Segment start in seconds.
        end_time: Segment end in seconds.
        sample_interval: Seconds between samples.
        max_samples: Maximum frames to analyze.

    Returns:
        List of face detections: [{'x': float, 'y': float, 'w': float, 'h': float, 'confidence': float}]
    """
    import cv2
    import mediapipe as mp

    mp_face = mp.solutions.face_detection
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    duration = end_time - start_time
    interval_frames = int(sample_interval * fps)
    total_frames = int(duration * fps)

    # Limit samples
    if total_frames // interval_frames > max_samples:
        interval_frames = total_frames // max_samples

    detections: List[Dict[str, Any]] = []

    with mp_face.FaceDetection(
        model_selection=1,  # 1 = full range, better for varying distances
        min_detection_confidence=0.5,
    ) as face_detection:

        frame_idx = 0
        sample_count = 0

        while cap.isOpened() and sample_count < max_samples:
            ret, frame = cap.read()
            if not ret:
                break

            current_time = start_time + (frame_idx / fps)

            if current_time > end_time:
                break

            if frame_idx % interval_frames == 0:
                h, w, _ = frame.shape
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_detection.process(rgb)

                if results.detections:
                    # Pick the largest/most confident face
                    best = max(results.detections, key=lambda d: d.location_data.relative_bounding_box.width)
                    bb = best.location_data.relative_bounding_box
                    detections.append({
                        "x": bb.xmin * w,
                        "y": bb.ymin * h,
                        "w": bb.width * w,
                        "h": bb.height * h,
                        "confidence": best.score[0],
                        "time": current_time,
                    })
                    sample_count += 1
                else:
                    # No face detected -- record frame center as fallback
                    detections.append({
                        "x": w * 0.25, "y": h * 0.1,
                        "w": w * 0.5, "h": h * 0.5,
                        "confidence": 0.0,
                        "time": current_time,
                    })
                    sample_count += 1

            frame_idx += 1

    cap.release()
    return detections


def compute_crop_from_faces(
    detections: List[Dict[str, Any]],
    output_width: int = 1080,
    output_height: int = 1920,
    source_width: int = 1920,
    source_height: int = 1080,
    padding: float = 1.5,
) -> Dict[str, float]:
    """Compute stable crop coordinates from face detections.

    Uses median face position across all samples for stability.
    Falls back to center-crop if no faces detected.

    Args:
        detections: List of face detections from detect_faces_in_segment.
        output_width: Target output width.
        output_height: Target output height.
        source_width: Source video width.
        source_height: Source video height.
        padding: Padding multiplier around face (1.5 = 50% extra).

    Returns:
        Dict with 'crop_x', 'crop_y', 'crop_w', 'crop_h' in source coordinates.
    """
    if not detections:
        # Fallback: center crop
        src_aspect = source_width / source_height
        out_aspect = output_width / output_height

        if src_aspect > out_aspect:
            crop_h = source_height
            crop_w = crop_h * out_aspect
        else:
            crop_w = source_width
            crop_h = crop_w / out_aspect

        return {
            "crop_x": (source_width - crop_w) / 2,
            "crop_y": (source_height - crop_h) / 2,
            "crop_w": crop_w,
            "crop_h": crop_h,
        }

    # Use only high-confidence detections for median
    good_detections = [d for d in detections if d["confidence"] > 0.3]
    if not good_detections:
        good_detections = detections

    # Median face center and size
    face_cx = sum(d["x"] + d["w"] / 2 for d in good_detections) / len(good_detections)
    face_cy = sum(d["y"] + d["h"] / 2 for d in good_detections) / len(good_detections)
    face_w = max(d["w"] for d in good_detections) * padding
    face_h = max(d["h"] for d in good_detections) * padding

    # Ensure crop maintains target aspect ratio
    target_aspect = output_width / output_height
    crop_w = face_w
    crop_h = crop_w / target_aspect

    # If crop is too tall, widen it
    if crop_h < face_h:
        crop_h = face_h
        crop_w = crop_h * target_aspect

    # Add extra vertical padding for subtitle area (bottom 15%)
    crop_h += source_height * 0.15

    # Clamp to source dimensions
    crop_w = min(crop_w, source_width)
    crop_h = min(crop_h, source_height)

    # Center crop on face, clamped to bounds
    crop_x = max(0, min(face_cx - crop_w / 2, source_width - crop_w))
    crop_y = max(0, min(face_cy - crop_h / 2, source_height - crop_h))

    return {
        "crop_x": round(crop_x, 1),
        "crop_y": round(crop_y, 1),
        "crop_w": round(crop_w, 1),
        "crop_h": round(crop_h, 1),
    }


def build_crop_filter(
    crop: Dict[str, float],
    output_width: int = 1080,
    output_height: int = 1920,
) -> str:
    """Build FFmpeg crop+scale filter string.

    Args:
        crop: Dict from compute_crop_from_faces.
        output_width: Target output width.
        output_height: Target output height.

    Returns:
        FFmpeg filter string like 'crop=720:1280:100:50,scale=1080:1920'
    """
    return (
        f"crop={int(crop['crop_w'])}:{int(crop['crop_h'])}:"
        f"{int(crop['crop_x'])}:{int(crop['crop_y'])},"
        f"scale={output_width}:{output_height}"
    )


def analyze_segment_faces(
    video_path: str,
    start_time: float,
    end_time: float,
    output_width: int = 1080,
    output_height: int = 1920,
    source_width: int = 1920,
    source_height: int = 1080,
) -> Dict[str, Any]:
    """Full pipeline: detect faces and compute optimal crop for a segment.

    Args:
        video_path: Path to video.
        start_time: Segment start.
        end_time: Segment end.
        output_width: Target output width.
        output_height: Target output height.
        source_width: Source video width.
        source_height: Source video height.

    Returns:
        Dict with 'crop_filter', 'crop_coords', 'face_count', 'avg_confidence'.
    """
    detections = detect_faces_in_segment(video_path, start_time, end_time)
    crop = compute_crop_from_faces(
        detections, output_width, output_height, source_width, source_height,
    )
    crop_filter = build_crop_filter(crop, output_width, output_height)

    face_count = len([d for d in detections if d["confidence"] > 0.3])
    avg_confidence = (
        sum(d["confidence"] for d in detections if d["confidence"] > 0.3) / max(face_count, 1)
    )

    return {
        "crop_filter": crop_filter,
        "crop_coords": crop,
        "face_count": face_count,
        "avg_confidence": round(avg_confidence, 3),
    }
