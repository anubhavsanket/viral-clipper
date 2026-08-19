"""
Shared configuration for ViralClipper AI.
Detects GPU availability and provides encoder/format settings used across all modules.
"""

import os
import sys
import subprocess
from dataclasses import dataclass, field
from typing import List, Optional


def detect_nvidia_gpu() -> bool:
    """Detect if NVIDIA GPU with NVENC support is available."""
    try:
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        subprocess.run(
            [
                "ffmpeg", "-f", "lavfi", "-i", "nullsrc",
                "-c:v", "h264_nvenc", "-t", "0.1",
                "-f", "null", "-",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return False


def get_base_dir() -> str:
    """Get the application base directory (works for PyInstaller and normal runs)."""
    if getattr(sys, "frozen", False):
        return sys._MEIPASS  # type: ignore[attr-defined]
    return os.path.dirname(os.path.abspath(__file__))


# --- Singleton-ish: detect once at import time ---
USE_GPU: bool = detect_nvidia_gpu()
VIDEO_ENCODER: str = "h264_nvenc" if USE_GPU else "libx264"
HANDBRAKE_ENCODER: str = "nvenc_h264" if USE_GPU else "x264"
BASE_DIR: str = get_base_dir()

# Supported video extensions
SUPPORTED_VIDEO_EXTENSIONS: List[str] = [".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv"]


@dataclass
class SubtitleStyle:
    """Hormozi-style subtitle configuration."""
    font_name: str = "Arial Black"
    font_size: int = 85
    primary_color: str = "&H0000FFFF"   # Yellow (ASS format: AABBGGRR)
    outline_color: str = "&H00000000"   # Black
    back_color: str = "&H00000000"      # Black
    bold: bool = True
    outline_thickness: int = 4
    shadow: int = 0
    alignment: int = 2                  # Bottom-center
    margin_v: int = 550
    play_res_x: int = 1080
    play_res_y: int = 1920
    max_words_per_line: int = 2
    max_chars_per_line: int = 18


@dataclass
class ClipConfig:
    """Configuration for clip rendering."""
    min_duration: float = 30.0
    max_duration: float = 179.0
    aspect_ratio: str = "9:16"          # Vertical short-form
    gpu_bitrate: str = "15M"
    cpu_crf: int = 18
    audio_bitrate: str = "192k"
    audio_codec: str = "aac"


@dataclass
class AnalysisConfig:
    """Configuration for LLM-based viral analysis."""
    default_model: str = "gemma3:4b"
    # Models to try if the primary model fails. Each entry is tried in order.
    fallback_models: List[str] = field(default_factory=lambda: [
        "qwen2.5:7b",
        "gemma3:4b",
        "llama3.1:8b",
        "mistral:7b",
        "phi4-mini:latest",
    ])
    default_prompt: str = (
        "Identify the most viral, funny, or engaging moments. "
        "Look for complete stories with setup, hook, and payoff."
    )
    min_duration: float = 30.0          # Minimum clip duration in seconds
    max_duration: float = 179.0         # Maximum clip duration in seconds
    target_clip_count: int = 5          # Number of clips to generate (1-20)
    target_duration: int = 90           # Preferred clip duration in seconds
    context_look_back: int = 2          # Max segments to look back
    min_gap_for_expansion: float = 1.5  # Seconds gap threshold for merging
    forward_gap_threshold: float = 1.0  # Seconds gap threshold for forward merge


@dataclass
class TranscriptionConfig:
    """Configuration for WhisperX transcription."""
    default_model_size: str = "small"
    compute_type: str = "int8"
    batch_size: int = 4
    device: str = "cpu"                 # Force CPU for Windows stability
    vad_filter: bool = False            # VAD can be too aggressive for music content
