"""
AI-powered viral clip analysis using Ollama LLM.
Identifies the most engaging segments from a video transcript.
Supports chunked analysis for long videos.
"""

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from config import AnalysisConfig


def _format_transcript_for_llm(segments: List[Dict[str, Any]]) -> str:
    """Format transcript segments into a timestamped string for the LLM.

    Args:
        segments: List of transcript segments with 'start', 'end', 'text'.

    Returns:
        Formatted transcript string.
    """
    lines: List[str] = []
    for seg in segments:
        start = round(seg["start"], 2)
        end = round(seg["end"], 2)
        text = seg["text"].strip()
        lines.append(f"[{start}-{end}] {text}")
    return "\n".join(lines)


def _extract_json_from_response(content: str) -> Optional[List[Dict[str, Any]]]:
    """Robustly extract JSON array from LLM response text.

    LLMs often wrap JSON in markdown fences, add explanatory text, etc.
    This function tries multiple strategies to extract valid JSON.

    Args:
        content: Raw LLM response string.

    Returns:
        Parsed list of clip dicts, or None if extraction fails.
    """
    # Strategy 1: Direct parse (if LLM returned clean JSON)
    try:
        result = json.loads(content)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # Strategy 2: Strip markdown code fences
    cleaned = content.replace("```json", "").replace("```", "").strip()
    try:
        result = json.loads(cleaned)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # Strategy 3: Find the first JSON array in the response using regex
    match = re.search(r"\[[\s\S]*?\]", cleaned)
    if match:
        try:
            result = json.loads(match.group(0))
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    # Strategy 4: Find the outermost array (handles nested objects)
    start_idx = cleaned.find("[")
    end_idx = cleaned.rfind("]")
    if start_idx != -1 and end_idx > start_idx:
        try:
            result = json.loads(cleaned[start_idx : end_idx + 1])
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    # Strategy 5: Heuristic extraction — pull timestamps from malformed output
    heuristic = _heuristic_extract_clips(content)
    if heuristic:
        return heuristic

    return None


def _heuristic_extract_clips(content: str) -> Optional[List[Dict[str, Any]]]:
    """Extract clip segments from malformed LLM output using pattern matching.

    Small models often produce garbled JSON with timestamps buried in text.
    This extracts any plausible start/end time pairs and builds clip dicts.

    Args:
        content: Raw LLM response string.

    Returns:
        List of clip dicts with start_time, end_time, reasoning; or None.
    """
    clips: List[Dict[str, Any]] = []

    # Pattern: "start_time": 12.0, "end_time": 145.0 (flexible)
    time_pattern = re.compile(
        r'"?start_time"?\s*[:=]\s*(\d+\.?\d*)\s*,?\s*"?end_?time"?\s*[:=]\s*(\d+\.?\d*)',
        re.IGNORECASE,
    )
    for m in time_pattern.finditer(content):
        start, end = float(m.group(1)), float(m.group(2))
        if end > start and (end - start) >= 10:
            clips.append({
                "start_time": start,
                "end_time": end,
                "virality_score": 50,
                "reasoning": "Extracted from malformed output",
            })

    # Pattern: timestamp ranges like "12.0 - 145.0" or "12.0-145.0"
    range_pattern = re.compile(r"(\d+\.?\d+)\s*[-–—to]+\s*(\d+\.?\d+)")
    for m in range_pattern.finditer(content):
        start, end = float(m.group(1)), float(m.group(2))
        if end > start and (end - start) >= 10 and (end - start) <= 300:
            # Avoid duplicates
            if not any(
                abs(c["start_time"] - start) < 1 and abs(c["end_time"] - end) < 1
                for c in clips
            ):
                clips.append({
                    "start_time": start,
                    "end_time": end,
                    "virality_score": 50,
                    "reasoning": "Extracted from malformed output",
                })

    # Pattern: MM:SS or M:SS timestamps
    ts_pattern = re.compile(r"(\d{1,2}):(\d{2})")
    ts_matches = list(ts_pattern.finditer(content))
    if len(ts_matches) >= 2 and not clips:
        for i in range(len(ts_matches) - 1):
            start_m, start_s = int(ts_matches[i].group(1)), int(ts_matches[i].group(2))
            end_m, end_s = int(ts_matches[i + 1].group(1)), int(ts_matches[i + 1].group(2))
            start = start_m * 60 + start_s
            end = end_m * 60 + end_s
            if end > start and (end - start) >= 10 and (end - start) <= 300:
                clips.append({
                    "start_time": float(start),
                    "end_time": float(end),
                    "virality_score": 50,
                    "reasoning": "Extracted from malformed output",
                })

    return clips if clips else None


def _estimate_tokens(text: str) -> int:
    """Rough token estimate (1 token ~ 4 chars for English)."""
    return len(text) // 4


def _split_transcript_into_chunks(
    segments: List[Dict[str, Any]],
    max_tokens: int = 6000,
    overlap_segments: int = 3,
) -> List[List[Dict[str, Any]]]:
    """Split transcript segments into overlapping chunks that fit within a context window.

    Args:
        segments: Full list of transcript segments.
        max_tokens: Maximum tokens per chunk (conservative estimate).
        overlap_segments: Number of segments to overlap between chunks.

    Returns:
        List of segment lists, each fitting within the token budget.
    """
    if not segments:
        return []

    # Estimate tokens for all text
    total_text = " ".join(seg.get("text", "") for seg in segments)
    total_tokens = _estimate_tokens(total_text)

    # If it fits in one chunk, return all segments as a single chunk
    if total_tokens <= max_tokens:
        return [segments]

    chunks: List[List[Dict[str, Any]]] = []
    current_chunk: List[Dict[str, Any]] = []
    current_tokens = 0
    header_overhead = 50  # Tokens for the prompt template itself

    for seg in segments:
        seg_tokens = _estimate_tokens(seg.get("text", ""))
        if current_tokens + seg_tokens > (max_tokens - header_overhead):
            if current_chunk:
                chunks.append(current_chunk)
                # Overlap: start next chunk from the last few segments
                current_chunk = current_chunk[-overlap_segments:] if len(current_chunk) > overlap_segments else []
                current_tokens = _estimate_tokens(" ".join(s.get("text", "") for s in current_chunk))
            current_chunk.append(seg)
            current_tokens += seg_tokens
        else:
            current_chunk.append(seg)
            current_tokens += seg_tokens

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def _build_analysis_prompt(
    transcript_text: str,
    user_prompt: str = "",
    chunk_info: str = "",
    target_clip_count: int = 5,
    target_duration: int = 90,
    simple: bool = False,
) -> str:
    """Build the LLM prompt for viral clip analysis.

    Args:
        transcript_text: The formatted transcript.
        user_prompt: Custom user instruction for what to look for.
        chunk_info: Information about which chunk this is (for multi-chunk analysis).
        target_clip_count: Desired number of clips to find.
        target_duration: Preferred clip duration in seconds.
        simple: If True, use a shorter prompt for small models.

    Returns:
        Complete prompt string.
    """
    instruction = user_prompt or (
        "Identify the most viral, funny, or engaging moments. "
        "Look for complete stories with setup, hook, and payoff."
    )

    if simple:
        return f"""Analyze this video transcript. Find {target_clip_count} viral clips (~{target_duration}s each, max {target_duration + 10}s).

CRITICAL: Return EXACTLY {target_clip_count} JSON objects in an array. Do NOT return fewer or more.
Do NOT read images or files. You only have this text.

Transcript:
{transcript_text}

Return ONLY a JSON array of exactly {target_clip_count} clips. No markdown, no explanation. Example:
[{{"start_time": 12.0, "end_time": 100.0, "virality_score": 90, "reasoning": "Funny moment about X"}},{{"start_time": 110.0, "end_time": 200.0, "virality_score": 85, "reasoning": "Another funny moment"}}]
"""

    return f"""You are an expert viral content editor. Analyze the transcript below from a video.
Your task: {instruction}

CRITICAL RULES:
1. Find EXACTLY {target_clip_count} clips. Not 1, not 0 — exactly {target_clip_count}.
2. EACH clip MUST be {target_duration} seconds long. Maximum allowed: {target_duration + 10} seconds. Shorter is OK, longer is NOT.
3. Include full context: setup, hook, and payoff.
4. No silence or filler clips.
5. DO NOT attempt to read images, screenshots, or files. You have NO visual access. You are a text-only model. Rely solely on the transcript text below.

{chunk_info}
Transcript:
{transcript_text}

Return ONLY a JSON array containing exactly {target_clip_count} objects. No markdown, no explanation:
[
  {{
    "start_time": 12.0,
    "end_time": 102.0,
    "virality_score": 95,
    "reasoning": "Complete story about X."
  }},
  {{
    "start_time": 115.0,
    "end_time": 205.0,
    "virality_score": 90,
    "reasoning": "Another engaging moment."
  }}
]
"""


def _smart_context_expansion(
    clips: List[Dict[str, Any]],
    segments: List[Dict[str, Any]],
    config: AnalysisConfig,
    target_duration: int = 90,
) -> List[Dict[str, Any]]:
    """Expand clip boundaries with surrounding context for better narrative flow.

    Clips are expanded toward target_duration but will NOT overlap with neighbors.

    Args:
        clips: Raw clips from LLM analysis.
        segments: Full transcript segments.
        config: Analysis configuration.
        target_duration: Target clip duration in seconds (caps expansion).

    Returns:
        Expanded clips with adjusted timestamps and durations.
    """
    if not segments:
        return clips

    max_time = segments[-1]["end"]
    # Hard cap: target + smaller buffer for tight duration control
    # For 30s: 30 + 8 = 38s, For 60s: 60 + 12 = 72s, For 90s: 90 + 15 = 105s
    buffer = min(int(target_duration * 0.2), 15)
    hard_cap = min(target_duration + buffer, config.max_duration)
    hard_cap = max(hard_cap, 20)

    print(f"--- Applying Smart Context Expansion (cap: {hard_cap}s) ---")

    # Sort clips by start time to know neighbors
    sorted_clips = sorted(clips, key=lambda c: c.get("start_time", 0))

    valid_clips: List[Dict[str, Any]] = []

    for idx, clip in enumerate(sorted_clips):
        original_start = clip.get("start_time", 0)
        original_end = clip.get("end_time", 0)

        # Determine neighbor boundaries (with 2s gap minimum)
        prev_end = sorted_clips[idx - 1]["end_time"] + 2 if idx > 0 else 0
        next_start = sorted_clips[idx + 1]["start_time"] - 2 if idx < len(sorted_clips) - 1 else max_time

        # Expansion cap: grow toward target + 5s buffer
        expand_cap = min(target_duration + 5, hard_cap)
        # Hard limit: don't expand past neighbors
        max_end = min(original_start + expand_cap, next_start)
        max_start = max(original_end - expand_cap, prev_end)

        # Find segment indices corresponding to LLM timestamps
        start_index = -1
        end_index = -1

        for i, seg in enumerate(segments):
            if start_index == -1 and seg["end"] >= original_start:
                start_index = i
            if seg["start"] <= original_end:
                end_index = i

        if start_index == -1:
            start_index = 0
        if end_index == -1:
            end_index = len(segments) - 1
        if end_index < start_index:
            end_index = start_index

        # Look back (add intro/setup) -- respect cap and neighbor boundary
        current_start = segments[start_index]["start"]
        steps_back = 0
        while start_index > 0 and steps_back < config.context_look_back:
            prev_seg = segments[start_index - 1]

            # Don't look back past previous clip's end
            if prev_seg["start"] < prev_end:
                break

            gap = current_start - prev_seg["end"]
            current_duration = segments[end_index]["end"] - current_start

            # Don't look back if it would exceed the cap
            if current_duration >= expand_cap:
                break

            if gap < config.min_gap_for_expansion or current_duration < 20:
                start_index -= 1
                current_start = prev_seg["start"]
                steps_back += 1
            else:
                break

        # Look forward (finish the thought) -- respect cap and neighbor boundary
        current_end = segments[end_index]["end"]
        while end_index < len(segments) - 1:
            next_seg = segments[end_index + 1]

            # Don't look forward past next clip's start
            if next_seg["start"] > next_start:
                break

            current_duration = current_end - segments[start_index]["start"]

            if current_duration >= expand_cap:
                break

            gap = next_seg["start"] - current_end
            if gap < config.forward_gap_threshold or current_duration < config.min_duration:
                end_index += 1
                current_end = next_seg["end"]
            else:
                break

        # Finalize timestamps
        final_start = segments[start_index]["start"]
        final_end = segments[end_index]["end"]

        # Clamp to video end
        if final_end > max_time:
            final_end = max_time

        # Clamp to hard cap
        if (final_end - final_start) > hard_cap:
            final_end = final_start + hard_cap

        # Clamp to neighbor boundaries (safety net)
        if final_end > next_start:
            final_end = next_start
        if final_start < prev_end:
            final_start = prev_end

        # Final duration sanity check
        if (final_end - final_start) < 5:
            print(f"  Skipping clip: too short after clamping ({final_end - final_start:.1f}s)")
            continue

        clip["start_time"] = round(final_start, 2)
        clip["end_time"] = round(final_end, 2)
        clip["duration"] = round(final_end - final_start, 2)

        print(
            f"  Clip: {original_start}-{original_end} -> "
            f"{clip['start_time']}-{clip['end_time']} "
            f"(Dur: {clip['duration']}s)"
        )
        valid_clips.append(clip)

    return valid_clips


def analyze_transcript(
    input_json: str,
    output_clips: str,
    model_name: Optional[str] = None,
    user_prompt: str = "",
    config: Optional[AnalysisConfig] = None,
    progress_callback=None,
    target_clip_count: Optional[int] = None,
    target_duration: Optional[int] = None,
) -> None:
    """Analyze transcript to find viral clip segments using Ollama LLM.

    Supports chunked analysis for long videos that exceed the LLM's context window.

    Args:
        input_json: Path to transcript JSON file.
        output_clips: Path to output clips JSON file.
        model_name: Ollama model name (e.g., 'gemma3:4b').
        user_prompt: Custom analysis instruction.
        config: Analysis configuration.
        progress_callback: Optional callback for progress updates (0-100).
        target_clip_count: Number of clips to find (overrides config).
        target_duration: Preferred clip duration in seconds (overrides config).

    Raises:
        FileNotFoundError: If input_json does not exist.
        ValueError: If LLM returns unparseable JSON.
    """
    if config is None:
        config = AnalysisConfig()
    if model_name is None:
        model_name = config.default_model

    print(f"--- Loading {input_json} ---")
    if not os.path.exists(input_json):
        raise FileNotFoundError(f"Transcript file not found: {input_json}")

    with open(input_json, "r", encoding="utf-8") as f:
        segments = json.load(f)

    if not segments:
        raise ValueError(
            "Transcript is empty. No segments to analyze. "
            "Check that the video has audible speech."
        )

    # Resolve clip count and duration (params override config)
    clip_count = target_clip_count or config.target_clip_count
    clip_duration = target_duration or config.target_duration

    # Chunk the transcript for context-window safety
    chunks = _split_transcript_into_chunks(segments, max_tokens=6000)
    total_chunks = len(chunks)
    all_raw_clips: List[Dict[str, Any]] = []

    print(f"--- Sending to Ollama ({model_name})... ---")
    print(f"--- Target: {clip_count} clips, ~{clip_duration}s each ---")
    if total_chunks > 1:
        print(f"--- Transcript split into {total_chunks} chunks for context safety ---")

    import ollama

    def _is_model_available(name: str) -> bool:
        """Check if a model is available locally in Ollama."""
        try:
            models = ollama.list()
            return any(m.get("name", "").startswith(name.split(":")[0]) for m in models.get("models", []))
        except Exception:
            return False

    def _try_pull_model(name: str) -> bool:
        """Attempt to pull a model. Returns True if successful."""
        try:
            print(f"  Pulling {name}... (this may take a few minutes)")
            ollama.pull(name)
            return True
        except Exception as e:
            print(f"  Failed to pull {name}: {e}")
            return False

    # Build model chain: primary + fallbacks that are available locally
    models_to_try = [model_name]
    for fb in config.fallback_models:
        if fb != model_name and _is_model_available(fb):
            models_to_try.append(fb)
    # If no fallbacks available, try pulling the first one
    if len(models_to_try) < 2 and config.fallback_models:
        for fb in config.fallback_models:
            if fb != model_name:
                if _try_pull_model(fb):
                    models_to_try.append(fb)
                break

    print(f"--- Model chain: {models_to_try} ---")

    for chunk_idx, chunk_segments in enumerate(chunks):
        transcript_text = _format_transcript_for_llm(chunk_segments)
        chunk_info = f"(Analyzing chunk {chunk_idx + 1}/{total_chunks})" if total_chunks > 1 else ""

        prompt = _build_analysis_prompt(
            transcript_text, user_prompt, chunk_info,
            target_clip_count=clip_count,
            target_duration=clip_duration,
        )

        clips_data = None
        for model in models_to_try:
            print(f"\n--- Querying LLM: {model} (chunk {chunk_idx + 1}/{total_chunks}) ---")
            for attempt in range(3):
                simple = attempt >= 1
                try:
                    response = ollama.chat(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        options={
                            "temperature": 0.3,
                            "num_predict": 2048,
                            "top_p": 0.9,
                            "repeat_penalty": 1.3,
                        },
                    )
                    content = response["message"]["content"]
                except Exception as e:
                    print(f"  Error calling {model}: {e}")
                    break

                print(f"  Response (attempt {attempt + 1}): {content[:300]}...")

                clips_data = _extract_json_from_response(content)
                if clips_data is not None:
                    break

                if attempt < 2:
                    prompt = _build_analysis_prompt(
                        transcript_text, user_prompt, chunk_info,
                        target_clip_count=clip_count,
                        target_duration=clip_duration,
                        simple=True,
                    )

            if clips_data is not None:
                print(f"  Success with {model}")
                break
            print(f"  {model} failed all attempts, trying next model...")

        if clips_data is None:
            print("Warning: All models failed for this chunk. Skipping.")
            continue

        # Validate clip structure
        valid_clips = []
        for clip in clips_data:
            if not isinstance(clip, dict):
                continue
            if "start_time" not in clip or "end_time" not in clip:
                continue
            try:
                clip["start_time"] = float(clip["start_time"])
                clip["end_time"] = float(clip["end_time"])
                valid_clips.append(clip)
            except (ValueError, TypeError):
                continue

        all_raw_clips.extend(valid_clips)

        # Report chunk progress if callback provided
        if progress_callback:
            progress_callback(int(50 * (chunk_idx + 1) / total_chunks))

    if not all_raw_clips:
        raise ValueError(
            "No valid clips found in LLM response. "
            "Try a different model or adjust the analysis prompt."
        )

    # Apply smart context expansion on full segment list
    # The hard_cap inside expansion will enforce the target duration
    expanded_clips = _smart_context_expansion(all_raw_clips, segments, config, target_duration=clip_duration)

    # Remove overlapping clips (keep the one with higher virality score)
    expanded_clips.sort(key=lambda c: c.get("virality_score", 0), reverse=True)
    deduplicated: List[Dict[str, Any]] = []
    for clip in expanded_clips:
        overlap = False
        for existing in deduplicated:
            if (
                clip["start_time"] < existing["end_time"]
                and clip["end_time"] > existing["start_time"]
            ):
                overlap = True
                break
        if not overlap:
            deduplicated.append(clip)

    # Sort by start time for chronological order
    deduplicated.sort(key=lambda c: c["start_time"])

    # Save results
    os.makedirs(os.path.dirname(output_clips) or ".", exist_ok=True)
    with open(output_clips, "w", encoding="utf-8") as f:
        json.dump(deduplicated, f, indent=2)

    print(f"\n--- Saved {len(deduplicated)} clips to {output_clips} ---")


if __name__ == "__main__":
    analyze_transcript("transcript.json", "clips.json")
