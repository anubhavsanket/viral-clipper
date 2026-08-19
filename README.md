# 🚀 ViralClipper AI

> **Your Local "OpusClip" Alternative.**  
> Automatically turn long videos into viral, captioned shorts using local AI. No cloud fees, total privacy.

---

## 🔥 Features
*   **Privacy First:** Runs 100% offline using `Ollama` and `WhisperX`.
*   **AI Virality:** Uses LLMs (Gemma/Llama) to find the funniest or most engaging hooks.
*   **Hormozi Captions:** Auto-generates punchy, word-level animated subtitles (yellow/white style).
*   **Smart Pipeline:**
    *   **Transcribe:** `WhisperX` (Fastest word-level timestamps).
    *   **Analyze:** `Ollama` (Context-aware clipping).
    *   **Edit:** `FFmpeg` & `HandBrake` (9:16 Crop + Burn-in).
*   **Dual Mode:**
    *   🖥️ **UI Dashboard:** Interactive drag-and-drop (`Start_App.bat`).
    *   ⚙️ **Batch Mode:** Process entire folders overnight (`5_batch_runner.py`).

---

## 🛠️ Requirements

### Hardware
*   **OS:** Windows 10/11
*   **GPU:** NVIDIA RTX recommended (for WhisperX & NVENC). *CPU mode is possible but slower.*
*   **RAM:** 16GB+ (for running Ollama models).

### Software
1.  **Python 3.10**: [Download Here](https://www.python.org/downloads/release/python-31011/)
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
    *Note: If you have issues with WhisperX/Torch, ensure you have the CUDA 11.8+ version of PyTorch.*

3.  **Setup AI Models**:
    Open a terminal and run:
    ```bash
    ollama pull gemma3:4b
    ```
    *(You can also pull `mistral` or `llama3` and select them in the UI).*

---

## 🎬 How to Run

### Option A: The Dashboard (Recommended)
Double-click **`Start_App.bat`**.
1.  Select your video file.
2.  Choose your AI models.
3.  Click **START PROCESSING**.

### Option B: Batch Mode (Headless)
1.  Place all your videos in `input_videos/`.
2.  Run the script:
    ```bash
    python 5_batch_runner.py
    ```
3.  Results will appear in `processed_videos/`.

---

## 📂 Project Structure

| File | Purpose |
| :--- | :--- |
| **`app_ui.py`** | The PyQt6 Graphical Interface. |
| **`pipeline_manager.py`** | Orchestrates the background threads and tool checks. |
| **`1_transcribe.py`** | WhisperX engine for audio-to-text. |
| **`2_analyze.py`** | Connects to Ollama to find viral clips. |
| **`3_process_clips.py`** | FFmpeg logic for cropping and rendering. |
| **`4_generate_report.py`** | Generates the HTML/Markdown report. |
| **`5_batch_runner.py`** | Legacy script for folder-based processing. |

---

## 🐛 Troubleshooting

*   **"Ollama not found"**: Ensure `ollama serve` is running in a separate terminal window.
*   **"HandBrakeCLI missing"**: Download the command-line version of HandBrake and place it in the project root or `bin/`.
*   **Crashes on Transcribe**: Decrease `BATCH_SIZE` in `1_transcribe.py` if running out of VRAM.
