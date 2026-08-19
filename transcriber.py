"""
Transcription engine using faster-whisper.
Handles speech-to-text with word-level timestamps and Hormozi-style subtitle generation.
Uses faster-whisper directly (no pyannote/speechbrain dependency).
"""

import gc
import json
import os
from typing import Any, Dict, List, Optional

from config import SubtitleStyle, TranscriptionConfig


def format_timestamp_ass(seconds: Optional[float]) -> str:
    """Convert seconds to ASS timestamp format (H:MM:SS.cs).

    Args:
        seconds: Time in seconds, or None for 0:00:00.00.

    Returns:
        ASS-formatted timestamp string.
    """
    if seconds is None:
        return "0:00:00.00"
    seconds = max(0.0, seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centis = int(round((seconds - int(seconds)) * 100))
    return f"{hours}:{minutes:02}:{secs:02}.{centis:02}"


def create_word_chunks(
    segments: List[Dict[str, Any]],
    style: Optional[SubtitleStyle] = None,
) -> List[Dict[str, Any]]:
    """Split long Whisper segments into punchy 2-3 word chunks for Hormozi-style captions.

    Args:
        segments: List of segments with word-level data.
        style: Subtitle style configuration (uses defaults if None).

    Returns:
        List of chunk dicts with 'text', 'start', and 'end' keys.
    """
    if style is None:
        style = SubtitleStyle()

    chunks: List[Dict[str, Any]] = []

    for segment in segments:
        if "words" not in segment:
            continue

        current_chunk: List[Dict[str, Any]] = []
        current_char_count = 0

        for word_obj in segment["words"]:
            word = word_obj.get("word", "").strip()
            start = word_obj.get("start")
            end = word_obj.get("end")

            if start is None or end is None:
                continue

            # Check if we should flush the current chunk
            if (
                len(current_chunk) >= style.max_words_per_line
                or (current_char_count + len(word)) > style.max_chars_per_line
            ):
                if current_chunk:
                    chunks.append({
                        "text": " ".join(w["word"] for w in current_chunk),
                        "start": current_chunk[0]["start"],
                        "end": current_chunk[-1]["end"],
                    })
                    current_chunk = []
                    current_char_count = 0

            current_chunk.append({"word": word, "start": start, "end": end})
            current_char_count += len(word) + 1

        # Flush remaining words
        if current_chunk:
            chunks.append({
                "text": " ".join(w["word"] for w in current_chunk).upper(),
                "start": current_chunk[0]["start"],
                "end": current_chunk[-1]["end"],
            })

    return chunks


def save_ass_hormozi(
    chunks: List[Dict[str, Any]],
    filename: str,
    style: Optional[SubtitleStyle] = None,
) -> None:
    """Generate a Hormozi-style ASS subtitle file.

    Args:
        chunks: Word chunks with 'text', 'start', 'end'.
        filename: Output .ass file path.
        style: Subtitle style configuration.
    """
    if style is None:
        style = SubtitleStyle()

    bold_val = "-1" if style.bold else "0"
    color_val = style.primary_color
    outline_val = str(style.outline_thickness)

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {style.play_res_x}
PlayResY: {style.play_res_y}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{style.font_name},{style.font_size},{color_val},&H000000FF,{style.outline_color},{style.back_color},{bold_val},0,0,0,100,100,0,0,1,{outline_val},{style.shadow},{style.alignment},10,10,{style.margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)

    with open(filename, "w", encoding="utf-8") as f:
        f.write(header)
        for chunk in chunks:
            start_ts = format_timestamp_ass(chunk["start"])
            end_ts = format_timestamp_ass(chunk["end"])
            text = chunk["text"]
            f.write(f"Dialogue: 0,{start_ts},{end_ts},Default,,0,0,0,,{text}\n")


def transcribe_video(
    video_path: str,
    output_json: str,
    output_ass: str,
    model_size: Optional[str] = None,
    subtitle_style: Optional[SubtitleStyle] = None,
    config: Optional[TranscriptionConfig] = None,
) -> None:
    """Transcribe a video and generate Hormozi-style subtitles.

    Uses faster-whisper directly for transcription with word-level timestamps.
    No pyannote/speechbrain dependency required.

    Args:
        video_path: Path to input video file.
        output_json: Path for transcript JSON output.
        output_ass: Path for ASS subtitle output.
        model_size: Whisper model size (base/small/medium/large).
        subtitle_style: Custom subtitle styling.
        config: Transcription configuration.

    Raises:
        FileNotFoundError: If video_path does not exist.
        RuntimeError: If transcription fails.
    """
    if config is None:
        config = TranscriptionConfig()
    if subtitle_style is None:
        subtitle_style = SubtitleStyle()
    if model_size is None:
        model_size = config.default_model_size

    device = config.device
    compute_type = config.compute_type

    print(f"--- Loading Whisper Model ({model_size} | {compute_type} | {device}) ---")

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    try:
        from faster_whisper import WhisperModel

        # Load model (downloads on first run)
        print("--- Downloading/Loading model (first run may take a few minutes) ---")
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        print("--- Model loaded successfully ---")

        print(f"--- Transcribing Audio from {video_path} ---")
        raw_segments, info = model.transcribe(
            video_path,
            word_timestamps=True,
            language=None,  # Auto-detect language
            vad_filter=config.vad_filter,
        )

        # Convert faster-whisper segments to our format
        segments: List[Dict[str, Any]] = []
        for seg in raw_segments:
            segment_dict: Dict[str, Any] = {
                "start": seg.start,
                "end": seg.end,
                "text": seg.text.strip(),
            }
            if seg.words:
                segment_dict["words"] = [
                    {
                        "word": w.word.strip(),
                        "start": w.start,
                        "end": w.end,
                    }
                    for w in seg.words
                ]
            segments.append(segment_dict)

        print(f"--- Transcription complete: {len(segments)} segments (lang={info.language}, prob={info.language_probability:.2f}) ---")

        # Free model
        print("--- Cleaning up model ---")
        del model
        gc.collect()

        # Validate transcript is not empty
        if not segments:
            raise RuntimeError(
                "Transcription produced no segments. "
                "The video may be silent or the audio quality too low."
            )

        # Save transcript JSON
        print(f"--- Saving to {output_json} ---")
        os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(segments, f, indent=2)

        # Generate word chunks and ASS subtitles
        print("--- Chunking words for viral style... ---")
        word_chunks = create_word_chunks(segments, style=subtitle_style)
        save_ass_hormozi(word_chunks, output_ass, style=subtitle_style)

        print(f"Done! Created {len(word_chunks)} punchy subtitle chunks.")

    except FileNotFoundError:
        raise
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"Transcription failed: {e}") from e


if __name__ == "__main__":
    VIDEO_PATH = "input_video.mp4"
    if not os.path.exists(VIDEO_PATH):
        print(f"Error: {VIDEO_PATH} not found.")
    else:
        transcribe_video(VIDEO_PATH, "transcript.json", "subtitles.ass")
