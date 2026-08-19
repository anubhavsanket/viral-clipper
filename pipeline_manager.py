"""
Pipeline orchestrator for ViralClipper AI.
Manages the full pipeline execution with progress tracking and stop-signal support.
"""

import os
import subprocess
import sys
import threading
import time
import traceback
from typing import Callable, Optional

import requests

from config import BASE_DIR, HANDBRAKE_ENCODER, USE_GPU

# Lazy imports for pipeline modules (avoids loading heavy deps at module level)
_transcriber = None
_analyzer = None
_clip_processor = None
_report_generator = None


def _ensure_modules_loaded():
    """Lazy-load pipeline modules on first use."""
    global _transcriber, _analyzer, _clip_processor, _report_generator

    if _transcriber is None:
        from transcriber import transcribe_video
        _transcriber = transcribe_video
    if _analyzer is None:
        from analyzer import analyze_transcript
        _analyzer = analyze_transcript
    if _clip_processor is None:
        from clip_processor import process_clips
        _clip_processor = process_clips
    if _report_generator is None:
        from report_generator import generate_report
        _report_generator = generate_report


class PipelineManager:
    """Manages the ViralClipper AI pipeline execution.

    Runs the full pipeline (pre-process, transcribe, analyze, render, report)
    on a background thread with progress callbacks and stop-signal support.
    """

    def __init__(
        self,
        log_callback: Optional[Callable[[str], None]] = None,
        progress_callback: Optional[Callable[[int], None]] = None,
    ):
        """Initialize the pipeline manager.

        Args:
            log_callback: Function to receive log messages.
            progress_callback: Function to receive progress updates (0-100).
        """
        self.log_callback = log_callback
        self.progress_callback = progress_callback
        self.should_stop = False
        self.ollama_process: Optional[subprocess.Popen] = None
        self.is_running = False

    def stop(self) -> None:
        """Signal the pipeline to stop."""
        self.should_stop = True
        self.log("Stop signal received. Pipeline will halt safely.")

    def log(self, message: str) -> None:
        """Send a log message to the callback and stdout."""
        print(message)
        if self.log_callback:
            self.log_callback(message)

    def _update_progress(self, value: int) -> None:
        """Send a progress update."""
        if self.progress_callback:
            self.progress_callback(value)

    def check_tools(self) -> tuple:
        """Check that required tools (FFmpeg, HandBrake, FFprobe) are available.

        Returns:
            Tuple of (success: bool, handbrake_path: Optional[str]).
        """
        tools = ["HandBrakeCLI.exe", "ffmpeg.exe", "ffprobe.exe"]
        missing = []

        search_paths = [
            os.getcwd(),
            os.path.join(os.getcwd(), "bin"),
            os.path.dirname(sys.argv[0]) if sys.argv[0] else os.getcwd(),
            os.path.join(os.path.dirname(sys.argv[0]) if sys.argv[0] else "", "bin"),
            BASE_DIR,
            os.path.join(BASE_DIR, "bin"),
        ]

        found_hb = None

        for tool in tools:
            found = False
            for path in search_paths:
                full_path = os.path.join(path, tool)
                if os.path.exists(full_path):
                    if tool == "HandBrakeCLI.exe":
                        found_hb = full_path
                    if path not in os.environ["PATH"]:
                        os.environ["PATH"] += os.pathsep + path
                    found = True
                    break
            if not found:
                missing.append(tool)

        if missing:
            self.log(f"Missing tools: {', '.join(missing)}")
            self.log("Please ensure they are in the 'bin' folder.")
            return False, None

        self.log(f"Found HandBrake at: {found_hb}")
        return True, found_hb

    def ensure_ollama_running(self) -> bool:
        """Check if Ollama is running, start it if not.

        Returns:
            True if Ollama is running, False otherwise.
        """
        try:
            requests.get("http://127.0.0.1:11434", timeout=2)
            self.log("Ollama is already running.")
            return True
        except requests.exceptions.ConnectionError:
            self.log("Ollama not detected. Starting background service...")
            try:
                creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
                self.ollama_process = subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=creationflags,
                )
                for _ in range(10):
                    try:
                        time.sleep(1)
                        requests.get("http://127.0.0.1:11434", timeout=1)
                        self.log("Ollama started successfully.")
                        return True
                    except requests.exceptions.ConnectionError:
                        pass
                self.log("Failed to start Ollama automatically.")
                return False
            except FileNotFoundError:
                self.log("'ollama' command not found. Please install Ollama.")
                return False

    def convert_video_handbrake(self, src: str, dest: str, hb_path: str) -> None:
        """Convert video with HandBrake to fix VFR and force CFR.

        Args:
            src: Source video path.
            dest: Destination video path.
            hb_path: Path to HandBrakeCLI executable.

        Raises:
            RuntimeError: If conversion fails.
        """
        self.log(f"Converting {os.path.basename(src)} with HandBrake ({HANDBRAKE_ENCODER})...")

        cmd = [
            hb_path, "--input", src, "--output", dest,
            "--format", "av_mp4",
            "--encoder", HANDBRAKE_ENCODER,
            "--quality", "20", "--cfr",
            "--aencoder", "copy",
            "--audio-fallback", "aac", "--ab", "192",
        ]

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )

            for line in process.stdout:
                if self.should_stop:
                    process.terminate()
                    raise RuntimeError("Process stopped by user.")

            process.wait()

            if process.returncode != 0:
                raise RuntimeError(f"HandBrake failed with code {process.returncode}")

            self.log("Conversion successful.")

        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"HandBrake conversion failed: {e}") from e

    def _run_clips_with_progress(
        self, input_video, clips_json, subtitles_ass, final_clips_dir,
    ):
        """Run clip rendering with per-clip progress reporting."""
        import json

        with open(clips_json, "r") as f:
            clips = json.load(f)

        total = len(clips) if clips else 1

        def clip_progress(clip_pct):
            # Map clip progress (0-100) into the 50-80% pipeline range
            pipeline_pct = 50 + int(30 * clip_pct / 100)
            self._update_progress(pipeline_pct)

        _clip_processor(
            input_video, clips_json, subtitles_ass,
            final_clips_dir, progress_callback=clip_progress,
        )

    def run_pipeline(
        self,
        video_path: str,
        model_name: str,
        prompt: str = "",
        output_dir: Optional[str] = None,
        whisper_model: str = "base",
        target_clip_count: int = 5,
        target_duration: int = 90,
        aspect_ratio: str = "9:16",
        enable_face_tracking: bool = True,
        enable_similarity_dedup: bool = True,
        enable_evaluation: bool = True,
    ) -> None:
        """Execute the full pipeline.

        Steps:
            0. Pre-process video with HandBrake (fix VFR)
            1. Transcribe with faster-whisper
            2. Analyze with Ollama LLM
            2.5 Deduplicate via embeddings (optional)
            3. Render clips with FFmpeg (face-tracking or center-crop)
            4. Generate report
            5. Evaluate quality (optional)

        Args:
            video_path: Path to the input video.
            model_name: Ollama model name for analysis.
            prompt: Custom analysis prompt for viral content detection.
            output_dir: Output directory (default: processed_output/ next to video).
            whisper_model: Whisper model size (base/small/medium/large).
            target_clip_count: Number of clips to generate.
            target_duration: Preferred clip duration in seconds.
            aspect_ratio: Output aspect ratio (9:16, 1:1, 16:9).
            enable_face_tracking: Use face-tracking crop instead of center-crop.
            enable_similarity_dedup: Remove near-duplicate clips via embeddings.
            enable_evaluation: Run quality evaluation on output clips.
        """
        if self.is_running:
            return
        self.is_running = True
        self.should_stop = False
        start_time = time.time()

        try:
            _ensure_modules_loaded()

            # Check tools
            valid, hb_path = self.check_tools()
            if not valid:
                raise FileNotFoundError("Required tools missing. Check logs.")

            # Ensure Ollama is running
            if not self.ensure_ollama_running():
                raise RuntimeError("Ollama service could not be started.")

            # Setup directories
            if not output_dir:
                output_dir = os.path.join(os.path.dirname(video_path), "processed_output")

            temp_dir = os.path.join(output_dir, "temp_work")
            os.makedirs(temp_dir, exist_ok=True)

            temp_input = os.path.join(temp_dir, "temp_input.mp4")
            transcript_json = os.path.join(temp_dir, "transcript.json")
            subtitles_ass = os.path.join(temp_dir, "subtitles.ass")
            clips_json = os.path.join(temp_dir, "clips.json")
            final_clips_dir = os.path.join(
                output_dir, os.path.splitext(os.path.basename(video_path))[0],
            )

            # Step 0: Pre-process
            if self.should_stop:
                return
            self.log("Step 0: Pre-processing Video...")
            self.convert_video_handbrake(video_path, temp_input, hb_path)
            self._update_progress(10)

            # Step 1: Transcribe
            if self.should_stop:
                return
            self.log(f"Step 1: Transcribing ({whisper_model})...")
            _transcriber(temp_input, transcript_json, subtitles_ass, model_size=whisper_model)
            self._update_progress(30)

            # Step 2: Analyze
            if self.should_stop:
                return
            self.log(f"Step 2: Analyzing for Viral Clips ({model_name})...")
            self.log(f"  Target: {target_clip_count} clips, ~{target_duration}s each")
            _analyzer(
                transcript_json, clips_json,
                model_name=model_name, user_prompt=prompt,
                progress_callback=lambda p: self._update_progress(50 + int(p * 0.05)),
                target_clip_count=target_clip_count,
                target_duration=target_duration,
            )
            self._update_progress(50)

            # Step 2.5: Deduplicate via embeddings
            if enable_similarity_dedup and not self.should_stop:
                self.log("Step 2.5: Removing duplicate clips (embeddings)...")
                try:
                    from clip_similarity import deduplicate_clips
                    import json as json_mod

                    with open(clips_json, "r", encoding="utf-8") as f:
                        clips_data = json_mod.load(f)

                    clips_data = deduplicate_clips(
                        clips_data, similarity_threshold=0.85, min_clips=target_clip_count,
                    )

                    with open(clips_json, "w", encoding="utf-8") as f:
                        json_mod.dump(clips_data, f, indent=2)

                    self.log(f"  Kept {len(clips_data)} unique clips after dedup.")
                except ImportError:
                    self.log("  Skipping dedup (sentence-transformers not installed).")
                except Exception as e:
                    self.log(f"  Dedup failed: {e}. Continuing without dedup.")
                self._update_progress(55)

            # Step 3: Render clips with per-clip progress
            if self.should_stop:
                return
            self.log("Step 3: Rendering Clips (FFmpeg)...")
            if enable_face_tracking:
                self.log("  Using face-tracking crop")
            else:
                self.log("  Using center crop")
            self._run_clips_with_progress(temp_input, clips_json, subtitles_ass, final_clips_dir)
            self._update_progress(80)

            # Step 4: Generate report
            if self.should_stop:
                return
            self.log("Step 4: Generating Report...")
            _report_generator(clips_json, transcript_json, final_clips_dir)
            self._update_progress(90)

            # Step 5: Evaluate quality
            if enable_evaluation and not self.should_stop:
                self.log("Step 5: Evaluating clip quality...")
                try:
                    from evaluator import evaluate_all_clips, generate_evaluation_report

                    eval_results = evaluate_all_clips(
                        final_clips_dir, clips_json, subtitles_ass, target_duration,
                    )
                    report_path = generate_evaluation_report(
                        eval_results, os.path.join(final_clips_dir, "EVALUATION.md"),
                    )
                    self.log(f"  Evaluation report: {report_path}")

                    # Log scores
                    for r in eval_results:
                        if "score" in r:
                            s = r["score"]
                            self.log(f"  Clip {r['clip_index']+1}: {s.overall:.0f}/100 (audio={s.audio_score:.0f}, visual={s.visual_score:.0f}, captions={s.caption_score:.0f})")
                except ImportError:
                    self.log("  Skipping evaluation (deps not installed).")
                except Exception as e:
                    self.log(f"  Evaluation failed: {e}.")
                self._update_progress(95)

            duration = int(time.time() - start_time)
            self.log(f"Pipeline Completed in {duration}s. Output: {final_clips_dir}")
            self._update_progress(100)

        except Exception as e:
            self.log(f"Pipeline Failed: {str(e)}")
            self.log(traceback.format_exc())
        finally:
            self.is_running = False

    def start_thread(
        self,
        video_path: str,
        model_name: str,
        prompt: str = "",
        output_dir: Optional[str] = None,
        whisper_model: str = "base",
        target_clip_count: int = 5,
        target_duration: int = 90,
        aspect_ratio: str = "9:16",
        enable_face_tracking: bool = True,
        enable_similarity_dedup: bool = True,
        enable_evaluation: bool = True,
    ) -> None:
        """Start the pipeline on a background daemon thread.

        Args:
            video_path: Path to the input video.
            model_name: Ollama model name.
            prompt: Custom analysis prompt.
            output_dir: Output directory.
            whisper_model: Whisper model size.
            target_clip_count: Number of clips to generate.
            target_duration: Preferred clip duration in seconds.
            aspect_ratio: Output aspect ratio (9:16, 1:1, 16:9).
            enable_face_tracking: Use face-tracking crop.
            enable_similarity_dedup: Remove duplicate clips via embeddings.
            enable_evaluation: Run quality evaluation.
        """
        if not self.is_running:
            t = threading.Thread(
                target=self.run_pipeline,
                args=(video_path, model_name, prompt, output_dir, whisper_model,
                      target_clip_count, target_duration, aspect_ratio,
                      enable_face_tracking, enable_similarity_dedup, enable_evaluation),
                daemon=True,
            )
            t.start()
        else:
            self.log("Pipeline is already running.")
