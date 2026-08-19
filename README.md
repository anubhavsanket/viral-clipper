# 🚀 ViralClipper AI

> **Your Local "OpusClip" Alternative.**  
> Automatically turn long videos into viral, captioned shorts using local AI. No cloud fees, total privacy.

---

## 🔥 Features
*   **Privacy First:** Runs 100% offline using `Ollama` and `faster-whisper`.
*   **AI Virality:** Uses LLMs (Gemma, Llama, Qwen, Phi) to find the funniest or most engaging hooks with chunked analysis support for long videos.
*   **Hormozi Captions:** Auto-generates punchy, word-level animated subtitles (yellow/white style).
*   **Smart Pipeline:**
    *   **Transcribe:** `faster-whisper` (Fast, stable word-level timestamps without pyannote/speechbrain hanging issues).
    *   **Analyze:** `Ollama` (Context-aware clipping).
    *   **Deduplicate:** `sentence-transformers` (Semantic embedding-based clip similarity check). [NEW]
    *   **Face Tracking:** `MediaPipe` (Intelligent face-tracking crop instead of static center-crop). [NEW]
    *   **Edit:** `FFmpeg` & `HandBrake` (9:16/1:1/16:9 Crop + Burn-in).
    *   **Evaluate:** Automated quality scoring (audio, visual, captions, duration). [NEW]
*   **Triple Mode:**
    *   🖥️ **UI Dashboard:** Interactive drag-and-drop PyQt6 GUI (`Start_App.bat`).
    *   🌐 **FastAPI Web Service:** REST API with background job processing and automatic evaluation endpoints (`api.py`). [NEW]
    *   ⚙️ **Batch Mode:** Process entire folders overnight (`batch_runner.py`).

---

## 🛠️ Requirements

### Hardware
- **OS:** Windows 10/11
- **GPU:** NVIDIA RTX recommended (for NVENC encoding). *CPU mode is possible but slower.*
- **RAM:** 16GB+ (for running Ollama models).

### Software
1.  **Python 3.10**: [Download Here](https://www.python.org/downloads/release/python-31011/) (Portable embedded Python included).
2.  **Ollama**: [Download Here](https://ollama.com/)
3.  **HandBrakeCLI**: Included in `bin/` or [Download Here](https://handbrake.fr/downloads.php).
4.  **FFmpeg**: Installed and valid in system PATH.

---

## 📦 Installation

1.  **Clone/Download** this repository.
2.  **Install Python Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: If you have issues with Torch, ensure you have the CUDA 11.8+ version of PyTorch.*

3.  **Setup AI Models**:
    Open a terminal and run:
    ```bash
    ollama pull gemma3:4b
    ```
    *(You can also pull `mistral`, `qwen2.5:7b`, or `llama3` and select them in the UI).*

---

## 🎬 How to Run

### Option A: The Dashboard (Recommended)
Double-click **`Start_App.bat`**.
1.  Select/drag your video file.
2.  Choose your AI models and clip settings (count, duration, aspect ratio, face-tracking, dedup, evaluation).
3.  Click **START PROCESSING**.

### Option B: Web API Service
Run the service:
```bash
python api.py
```
- Interactive API Docs at `http://localhost:8000/docs`
- Submit job: `POST /api/v1/generate`
- Check status: `GET /api/v1/jobs/{id}`

### Option C: Batch Mode (Headless)
1.  Place all your videos in `input_videos/`.
2.  Run the script:
    ```bash
    python batch_runner.py
    ```
3.  Results will appear in `processed_videos/`.

---

## 📂 Project Structure

| File | Purpose |
| :--- | :--- |
| **`app_ui.py`** | The PyQt6 Graphical Interface. |
| **`pipeline_manager.py`** | Orchestrates background thread pipeline execution. |
| **`transcriber.py`** | `faster-whisper` transcription engine. |
| **`analyzer.py`** | Ollama LLM viral clip analyzer with context safety checks. |
| **`clip_processor.py`** | FFmpeg crop, subtitle burn-in, and render engine. |
| **`face_tracker.py`** | MediaPipe face detection and tracking. |
| **`clip_similarity.py`** | Embedding-based clip similarity and deduplication. |
| **`evaluator.py`** | Multi-metric clip quality assessment. |
| **`report_generator.py`** | Virality timeline chart and markdown report generator. |
| **`api.py`** | FastAPI REST service with background queue. |
| **`batch_runner.py`** | Overnight folder-based video processing. |
| **`config.py`** | Shared config parameters and dataclasses. |
| **`tests/`** | 24 core pipeline unit tests (`pytest`). |

---

## 🐛 Troubleshooting

*   **"Ollama not found"**: Ensure Ollama is installed and running (`ollama serve`).
*   **"HandBrakeCLI missing"**: HandBrake is bundled in `bin/`. If missing, place `HandBrakeCLI.exe` there.
*   **Slow Transcription**: The embedded Python uses CPU-only torch. For GPU acceleration on transcription, install `torch` with CUDA support in your environment.
