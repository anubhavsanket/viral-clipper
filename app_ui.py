"""
ViralClipper AI - PyQt6 Desktop GUI.
Dark-themed dashboard for video clip generation with real-time progress tracking.
"""

import importlib.util
import os
import subprocess
import sys
from typing import Optional

# --- PATH FIX FOR PORTABLE PYTHON ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)


# --- DEPENDENCY AUTO-INSTALLER ---
def install_dependencies() -> None:
    """Check for GUI libs and install them to the embedded python."""
    required = ["PyQt6", "requests", "pillow"]
    optional = ["sentence-transformers", "numpy", "mediapipe", "opencv-python-headless", "fastapi", "uvicorn"]
    missing = []

    for lib in required:
        if importlib.util.find_spec(lib) is None:
            missing.append(lib)

    missing_optional = []
    for lib in optional:
        if importlib.util.find_spec(lib) is None:
            missing_optional.append(lib)

    if missing:
        print(f"First Run: Installing UI libraries ({', '.join(missing)})...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                "--upgrade", "pip", "--no-warn-script-location",
            ])
            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                "--prefer-binary", *missing, "--no-warn-script-location",
            ])
            print("Dependencies installed. Launching UI...")
        except Exception as e:
            print(f"Failed to install dependencies: {e}")
            input("Press Enter to exit...")
            sys.exit(1)

    if missing_optional:
        print(f"Installing optional features ({', '.join(missing_optional)})...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                "--prefer-binary", *missing_optional, "--no-warn-script-location",
            ])
            print("Optional features installed.")
        except Exception as e:
            print(f"Warning: Some optional features may not work: {e}")


if __name__ == "__main__":
    install_dependencies()


# --- IMPORTS ---
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QPushButton, QFileDialog,
    QProgressBar, QTextEdit, QComboBox, QFrame, QMessageBox,
    QLineEdit, QSplitter, QSlider, QSpinBox, QGroupBox, QCheckBox,
)
from PyQt6.QtCore import Qt, pyqtSignal, QObject, QTimer
from PyQt6.QtGui import QFont, QPalette, QColor, QDragEnterEvent, QDropEvent

from config import SUPPORTED_VIDEO_EXTENSIONS, SubtitleStyle

# Import backend safely
try:
    import pipeline_manager
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import pipeline_manager.\nError: {e}")
    input("Press Enter to exit...")
    sys.exit(1)


# --- STYLE CONSTANTS ---
COLORS = {
    "bg": "#191919",
    "surface": "#2D2D2D",
    "surface_light": "#333333",
    "border": "#444444",
    "border_light": "#555555",
    "primary": "#007ACC",
    "primary_hover": "#0069B4",
    "success": "#00C853",
    "success_hover": "#00B248",
    "danger": "#D50000",
    "danger_hover": "#B71C1C",
    "text": "#E0E0E0",
    "text_muted": "#AAAAAA",
    "text_dim": "#888888",
    "text_input": "#CCCCCC",
    "log_bg": "#0A0A0A",
    "log_text": "#00E676",
}

VIDEO_FILTER = "Video Files ({})".format(
    " ".join(f"*{ext}" for ext in SUPPORTED_VIDEO_EXTENSIONS)
)


# --- SIGNALS CLASS ---
class StreamSignals(QObject):
    """Thread-safe Qt signals for updating UI from background threads."""
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)


# --- MAIN UI CLASS ---
class ViralClipperUI(QMainWindow):
    """Main application window for ViralClipper AI."""

    def __init__(self):
        super().__init__()

        self.setWindowTitle("ViralClipper AI - Pro Dashboard")
        self.resize(1100, 880)

        # Initialize state
        self.selected_file: Optional[str] = None
        self.output_folder: Optional[str] = None

        # Build UI
        self.setup_ui()
        self.apply_dark_theme()

        # Thread-safe signals
        self.signals = StreamSignals()
        self.signals.log_signal.connect(self.append_log)
        self.signals.progress_signal.connect(self.update_progress)

        # Pipeline manager
        self.pipeline = pipeline_manager.PipelineManager(
            log_callback=self.signals.log_signal.emit,
            progress_callback=self.signals.progress_signal.emit,
        )

        self.append_log("System Ready.")
        self.append_log(f"Root Directory: {current_dir}")
        self.append_log(f"GPU Detected: {'Yes' if __import__('config').USE_GPU else 'No'}")

        # Fetch Ollama models AFTER the window is visible
        QTimer.singleShot(200, self.fetch_ollama_models)

    def setup_ui(self) -> None:
        """Build the complete UI layout."""
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setSpacing(12)
        main_layout.setContentsMargins(30, 25, 30, 25)

        # Enable drag-and-drop on main window
        self.setAcceptDrops(True)

        # --- HEADER ---
        header = QLabel("Viral Clipper AI Pipeline")
        header.setFont(QFont("Segoe UI", 20, QFont.Weight.Bold))
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setStyleSheet(f"color: {COLORS['text']}; margin-bottom: 15px;")
        main_layout.addWidget(header)

        # --- INPUT ROW ---
        input_layout = QHBoxLayout()
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Select or drag a video file...")
        self.input_field.setFixedHeight(42)
        self.input_field.setReadOnly(True)
        self.input_field.setStyleSheet(
            f"background-color: {COLORS['surface']}; color: {COLORS['text_input']}; "
            f"border: 1px solid {COLORS['border']}; border-radius: 5px; padding: 5px 10px;"
        )

        browse_input = QPushButton("Browse Input")
        browse_input.setFixedSize(130, 42)
        browse_input.setCursor(Qt.CursorShape.PointingHandCursor)
        browse_input.setStyleSheet(self._button_style(COLORS["primary"], COLORS["primary_hover"]))
        browse_input.clicked.connect(self.select_video)

        input_layout.addWidget(self.input_field)
        input_layout.addWidget(browse_input)
        main_layout.addLayout(input_layout)

        # --- OUTPUT ROW ---
        output_layout = QHBoxLayout()
        self.output_field = QLineEdit()
        self.output_field.setPlaceholderText("Select output folder (optional)...")
        self.output_field.setFixedHeight(42)
        self.output_field.setReadOnly(True)
        self.output_field.setStyleSheet(
            f"background-color: {COLORS['surface']}; color: {COLORS['text_input']}; "
            f"border: 1px solid {COLORS['border']}; border-radius: 5px; padding: 5px 10px;"
        )

        browse_output = QPushButton("Browse Output")
        browse_output.setFixedSize(130, 42)
        browse_output.setCursor(Qt.CursorShape.PointingHandCursor)
        browse_output.setStyleSheet(self._button_style(COLORS["primary"], COLORS["primary_hover"]))
        browse_output.clicked.connect(self.select_output)

        output_layout.addWidget(self.output_field)
        output_layout.addWidget(browse_output)
        main_layout.addLayout(output_layout)

        # --- ANALYSIS PROMPT ROW ---
        prompt_layout = QHBoxLayout()
        prompt_label = QLabel("Analysis Prompt:")
        prompt_label.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        prompt_label.setStyleSheet(f"color: {COLORS['text']};")

        self.prompt_field = QLineEdit()
        self.prompt_field.setPlaceholderText("Identify the most viral, funny, or engaging moments...")
        self.prompt_field.setFixedHeight(42)
        self.prompt_field.setStyleSheet(
            f"background-color: {COLORS['surface']}; color: {COLORS['text_input']}; "
            f"border: 1px solid {COLORS['border']}; border-radius: 5px; padding: 5px 10px;"
        )
        self.prompt_field.setText("Identify the most viral, funny, or engaging moments. Look for complete stories with setup, hook, and payoff.")

        prompt_layout.addWidget(prompt_label)
        prompt_layout.addWidget(self.prompt_field)
        main_layout.addLayout(prompt_layout)

        # --- SETTINGS ROW ---
        settings_layout = QHBoxLayout()
        settings_layout.setSpacing(15)
        settings_layout.setContentsMargins(0, 5, 0, 5)

        lbl_whisper = QLabel("Whisper Model:")
        lbl_whisper.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        lbl_whisper.setStyleSheet(f"color: {COLORS['text']};")
        self.whisper_combo = QComboBox()
        self.whisper_combo.addItems(["base", "small", "medium", "large"])
        self.whisper_combo.setFixedWidth(110)
        self.whisper_combo.setStyleSheet(self._combo_style())

        lbl_llm = QLabel("AI Model:")
        lbl_llm.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        lbl_llm.setStyleSheet(f"color: {COLORS['text']};")
        self.llm_combo = QComboBox()
        self.llm_combo.setEditable(True)
        self.llm_combo.setFixedWidth(160)
        self.llm_combo.setStyleSheet(self._combo_style())

        settings_layout.addWidget(lbl_whisper)
        settings_layout.addWidget(self.whisper_combo)
        settings_layout.addSpacing(20)
        settings_layout.addWidget(lbl_llm)
        settings_layout.addWidget(self.llm_combo)
        settings_layout.addStretch()

        main_layout.addLayout(settings_layout)

        # --- CLIP SETTINGS ROW ---
        clip_group = QGroupBox("Clip Settings")
        clip_group.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        clip_group.setStyleSheet(f"""
            QGroupBox {{
                color: {COLORS['text']};
                border: 1px solid {COLORS['border']};
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 15px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 5px;
            }}
        """)
        clip_layout = QHBoxLayout(clip_group)
        clip_layout.setSpacing(20)

        # Clip Count
        clip_count_layout = QVBoxLayout()
        lbl_clip_count = QLabel("Number of Clips:")
        lbl_clip_count.setStyleSheet(f"color: {COLORS['text_muted']};")
        clip_count_row = QHBoxLayout()
        self.clip_count_slider = QSlider(Qt.Orientation.Horizontal)
        self.clip_count_slider.setRange(1, 20)
        self.clip_count_slider.setValue(5)
        self.clip_count_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.clip_count_slider.setTickInterval(1)
        self.clip_count_slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                height: 6px; background: {COLORS['surface_light']}; border-radius: 3px;
            }}
            QSlider::handle:horizontal {{
                background: {COLORS['primary']}; width: 16px; height: 16px;
                margin: -5px 0; border-radius: 8px;
            }}
            QSlider::sub-page:horizontal {{
                background: {COLORS['primary']}; border-radius: 3px;
            }}
        """)
        self.clip_count_label = QLabel("5")
        self.clip_count_label.setFixedWidth(25)
        self.clip_count_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.clip_count_label.setStyleSheet(f"color: {COLORS['text']}; font-weight: bold;")
        self.clip_count_slider.valueChanged.connect(
            lambda v: self.clip_count_label.setText(str(v))
        )
        clip_count_row.addWidget(self.clip_count_slider)
        clip_count_row.addWidget(self.clip_count_label)
        clip_count_layout.addWidget(lbl_clip_count)
        clip_count_layout.addLayout(clip_count_row)
        clip_layout.addLayout(clip_count_layout)

        # Clip Duration
        clip_duration_layout = QVBoxLayout()
        lbl_duration = QLabel("Clip Duration:")
        lbl_duration.setStyleSheet(f"color: {COLORS['text_muted']};")
        self.duration_combo = QComboBox()
        self.duration_combo.addItems(["30s", "60s", "90s", "120s", "180s"])
        self.duration_combo.setCurrentText("90s")
        self.duration_combo.setFixedWidth(100)
        self.duration_combo.setStyleSheet(self._combo_style())
        clip_duration_layout.addWidget(lbl_duration)
        clip_duration_layout.addWidget(self.duration_combo)
        clip_layout.addLayout(clip_duration_layout)

        # Aspect Ratio
        aspect_layout = QVBoxLayout()
        lbl_aspect = QLabel("Aspect Ratio:")
        lbl_aspect.setStyleSheet(f"color: {COLORS['text_muted']};")
        self.aspect_combo = QComboBox()
        self.aspect_combo.addItems(["9:16 (Vertical)", "1:1 (Square)", "16:9 (Horizontal)"])
        self.aspect_combo.setFixedWidth(150)
        self.aspect_combo.setStyleSheet(self._combo_style())
        aspect_layout.addWidget(lbl_aspect)
        aspect_layout.addWidget(self.aspect_combo)
        clip_layout.addLayout(aspect_layout)

        clip_layout.addStretch()
        main_layout.addWidget(clip_group)

        # --- FEATURE TOGGLES ---
        features_layout = QHBoxLayout()
        features_layout.setSpacing(20)

        self.face_tracking_cb = QCheckBox("Face Tracking")
        self.face_tracking_cb.setChecked(True)
        self.face_tracking_cb.setToolTip("Track faces for intelligent cropping instead of center-crop")
        self.face_tracking_cb.setStyleSheet(f"color: {COLORS['text']}; spacing: 5px;")

        self.dedup_cb = QCheckBox("Deduplicate Clips")
        self.dedup_cb.setChecked(True)
        self.dedup_cb.setToolTip("Remove near-duplicate clips using embedding similarity")
        self.dedup_cb.setStyleSheet(f"color: {COLORS['text']}; spacing: 5px;")

        self.evaluate_cb = QCheckBox("Evaluate Quality")
        self.evaluate_cb.setChecked(True)
        self.evaluate_cb.setToolTip("Auto-score clips on audio/visual/caption quality")
        self.evaluate_cb.setStyleSheet(f"color: {COLORS['text']}; spacing: 5px;")

        features_layout.addWidget(self.face_tracking_cb)
        features_layout.addWidget(self.dedup_cb)
        features_layout.addWidget(self.evaluate_cb)
        features_layout.addStretch()
        main_layout.addLayout(features_layout)

        # --- ACTION BUTTONS ---
        btn_layout = QHBoxLayout()

        self.start_btn = QPushButton("START PROCESSING")
        self.start_btn.setFixedHeight(55)
        self.start_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.start_btn.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
        self.start_btn.setStyleSheet(self._button_style(COLORS["success"], COLORS["success_hover"]))
        self.start_btn.clicked.connect(self.start_pipeline)

        self.stop_btn = QPushButton("STOP")
        self.stop_btn.setFixedHeight(55)
        self.stop_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.stop_btn.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
        self.stop_btn.setStyleSheet(self._button_style(COLORS["danger"], COLORS["danger_hover"]))
        self.stop_btn.clicked.connect(self.stop_pipeline)
        self.stop_btn.setEnabled(False)

        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        main_layout.addLayout(btn_layout)

        # --- PROGRESS BAR ---
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(10)
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                border: none; background-color: {COLORS['surface_light']};
                border-radius: 5px;
                color: white;
                font-size: 11px;
                font-weight: bold;
                text-align: center;
            }}
            QProgressBar::chunk {{
                background-color: #4CAF50; border-radius: 5px;
            }}
        """)
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)

        # --- LOGS ---
        lbl_logs = QLabel("System Logs:")
        lbl_logs.setFont(QFont("Segoe UI", 10))
        lbl_logs.setStyleSheet(f"color: {COLORS['text_muted']}; margin-top: 8px;")
        main_layout.addWidget(lbl_logs)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setStyleSheet(f"""
            QTextEdit {{
                background-color: {COLORS['log_bg']};
                color: {COLORS['log_text']};
                font-family: Consolas;
                font-size: 10pt;
                border: 1px solid {COLORS['surface_light']};
                border-radius: 3px;
            }}
        """)
        main_layout.addWidget(self.log_output)

    def apply_dark_theme(self) -> None:
        """Apply dark palette to the application."""
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor(25, 25, 25))
        palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
        self.setPalette(palette)

    @staticmethod
    def _button_style(bg: str, hover: str) -> str:
        """Generate a QPushButton stylesheet."""
        return (
            f"QPushButton {{ background-color: {bg}; color: white; border-radius: 5px; font-weight: bold; }}"
            f"QPushButton:hover {{ background-color: {hover}; }}"
            f"QPushButton:disabled {{ background-color: {COLORS['surface']}; color: {COLORS['text_dim']}; }}"
        )

    @staticmethod
    def _combo_style() -> str:
        """Generate a QComboBox stylesheet."""
        return (
            f"padding: 5px; background-color: {COLORS['surface_light']}; "
            f"color: white; border: 1px solid {COLORS['border_light']};"
        )

    # --- OLLAMA MODEL DETECTION ---
    def fetch_ollama_models(self) -> None:
        """Fetch available models from Ollama and populate the combo box."""
        self.append_log("Scanning for local Ollama models...")
        try:
            creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True, text=True,
                creationflags=creationflags,
            )

            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                models = []
                # Skip header row (NAME, ID, SIZE, MODIFIED)
                for line in lines[1:]:
                    parts = line.split()
                    if parts:
                        models.append(parts[0])

                if models:
                    self.llm_combo.clear()
                    self.llm_combo.addItems(models)
                    self.append_log(f"Found {len(models)} local AI models.")
                else:
                    self.append_log("No models found in Ollama. Using defaults.")
            else:
                self.append_log("Could not contact Ollama. Is it running?")

        except FileNotFoundError:
            self.append_log("'ollama' command not found. Please install Ollama.")
        except Exception as e:
            self.append_log(f"Error fetching models: {e}")

    # --- DRAG AND DROP ---
    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        """Accept drag events that contain file URLs."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        """Handle dropped files."""
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            ext = os.path.splitext(file_path)[1].lower()
            if ext in SUPPORTED_VIDEO_EXTENSIONS:
                self.selected_file = file_path
                self.input_field.setText(file_path)
                self.append_log(f"Input (dragged): {os.path.basename(file_path)}")
            else:
                QMessageBox.warning(
                    self, "Unsupported Format",
                    f"File type '{ext}' is not supported.\n"
                    f"Supported: {', '.join(SUPPORTED_VIDEO_EXTENSIONS)}",
                )

    # --- FILE SELECTION ---
    def select_video(self) -> None:
        """Open file dialog to select a video file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video", "", VIDEO_FILTER,
        )
        if file_path:
            self.selected_file = file_path
            self.input_field.setText(file_path)
            self.append_log(f"Input Selected: {os.path.basename(file_path)}")

    def select_output(self) -> None:
        """Open folder dialog to select an output directory."""
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.output_folder = folder
            self.output_field.setText(folder)
            self.append_log(f"Output Folder Set: {folder}")

    # --- PIPELINE CONTROL ---
    def start_pipeline(self) -> None:
        """Start the clip generation pipeline."""
        if not self.selected_file:
            QMessageBox.warning(self, "No File", "Please select a video file first!")
            return

        if not os.path.exists(self.selected_file):
            QMessageBox.warning(
                self, "File Not Found",
                f"The selected file no longer exists:\n{self.selected_file}",
            )
            return

        whisper_model = self.whisper_combo.currentText()
        llm_model = self.llm_combo.currentText().strip()
        prompt = self.prompt_field.text().strip()

        # Read clip settings
        target_clip_count = self.clip_count_slider.value()
        duration_text = self.duration_combo.currentText().replace("s", "")
        target_duration = int(duration_text)

        # Parse aspect ratio
        aspect_text = self.aspect_combo.currentText()
        if "9:16" in aspect_text:
            aspect_ratio = "9:16"
        elif "1:1" in aspect_text:
            aspect_ratio = "1:1"
        else:
            aspect_ratio = "16:9"

        # Read feature toggles
        enable_face_tracking = self.face_tracking_cb.isChecked()
        enable_dedup = self.dedup_cb.isChecked()
        enable_eval = self.evaluate_cb.isChecked()

        target_dir = self.output_folder or os.path.dirname(self.selected_file)

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.log_output.clear()

        self.append_log("Starting Pipeline...")
        self.append_log(f"  Input: {os.path.basename(self.selected_file)}")
        self.append_log(f"  Output: {target_dir}")
        self.append_log(f"  Models: Whisper({whisper_model}) + LLM({llm_model})")
        self.append_log(f"  Clips: {target_clip_count} x ~{target_duration}s ({aspect_ratio})")
        self.append_log(f"  Features: face_track={enable_face_tracking}, dedup={enable_dedup}, eval={enable_eval}")

        self.pipeline.start_thread(
            video_path=self.selected_file,
            model_name=llm_model,
            prompt=prompt,
            output_dir=target_dir,
            whisper_model=whisper_model,
            target_clip_count=target_clip_count,
            target_duration=target_duration,
            aspect_ratio=aspect_ratio,
            enable_face_tracking=enable_face_tracking,
            enable_similarity_dedup=enable_dedup,
            enable_evaluation=enable_eval,
        )

    def stop_pipeline(self) -> None:
        """Signal the pipeline to stop."""
        self.pipeline.stop()
        self.append_log("Stopping pipeline... please wait.")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    # --- LOGGING & PROGRESS ---
    def append_log(self, message: str) -> None:
        """Append a message to the log window and auto-scroll."""
        self.log_output.append(message)
        cursor = self.log_output.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.log_output.setTextCursor(cursor)

        # Re-enable buttons on pipeline completion or failure
        if "Pipeline Completed" in message or "Pipeline Failed" in message:
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)

    def update_progress(self, value: int) -> None:
        """Update the progress bar."""
        self.progress_bar.setValue(value)


def main() -> None:
    """Application entry point."""
    app = QApplication(sys.argv)
    window = ViralClipperUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
