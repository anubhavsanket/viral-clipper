"""
ViralClipper AI - Web API Service.
FastAPI-based REST API for video clip generation with background task processing.
"""

import os
import sys
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

# --- Path setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from config import SUPPORTED_VIDEO_EXTENSIONS, AnalysisConfig

# --- App ---
app = FastAPI(
    title="ViralClipper AI",
    description="API for AI-powered viral clip generation from long-form videos.",
    version="2.0.0",
)

# --- In-memory job store (replace with Redis in production) ---
jobs: Dict[str, Dict[str, Any]] = {}


# --- Request/Response Models ---
class ClipRequest(BaseModel):
    """Request body for clip generation."""
    video_path: str = Field(..., description="Path to the input video file")
    output_dir: Optional[str] = Field(None, description="Output directory (default: auto)")
    model_name: str = Field("gemma3:4b", description="Ollama model for analysis")
    whisper_model: str = Field("base", description="Whisper model size")
    prompt: str = Field(
        "Identify the most viral, funny, or engaging moments.",
        description="Custom analysis prompt",
    )
    target_clip_count: int = Field(5, ge=1, le=20, description="Number of clips to generate")
    target_duration: int = Field(90, ge=10, le=180, description="Target clip duration (seconds)")
    aspect_ratio: str = Field("9:16", description="Output aspect ratio (9:16, 1:1, 16:9)")
    enable_face_tracking: bool = Field(True, description="Use face-tracking crop instead of center-crop")
    enable_similarity_dedup: bool = Field(True, description="Remove near-duplicate clips via embeddings")
    enable_evaluation: bool = Field(True, description="Run quality evaluation on output clips")


class JobStatus(BaseModel):
    """Job status response."""
    job_id: str
    status: str  # pending, running, completed, failed
    progress: int  # 0-100
    message: str
    created_at: str
    completed_at: Optional[str] = None
    output_dir: Optional[str] = None
    clips: Optional[list] = None
    evaluation: Optional[dict] = None


# --- Background task ---
def run_pipeline_job(job_id: str, request: ClipRequest) -> None:
    """Execute the full pipeline in background."""
    import traceback

    job = jobs[job_id]
    job["status"] = "running"
    job["message"] = "Pipeline starting..."

    try:
        # Lazy imports to avoid slow startup
        from pipeline_manager import PipelineManager

        def log_callback(msg: str) -> None:
            job["message"] = msg

        def progress_callback(val: int) -> None:
            job["progress"] = val

        manager = PipelineManager(
            log_callback=log_callback,
            progress_callback=progress_callback,
        )

        output_dir = request.output_dir or os.path.join(
            os.path.dirname(request.video_path), "viralclipper-output",
        )

        manager.run_pipeline(
            video_path=request.video_path,
            model_name=request.model_name,
            prompt=request.prompt,
            output_dir=output_dir,
            whisper_model=request.whisper_model,
            target_clip_count=request.target_clip_count,
            target_duration=request.target_duration,
            aspect_ratio=request.aspect_ratio,
        )

        job["status"] = "completed"
        job["progress"] = 100
        job["completed_at"] = datetime.now().isoformat()
        job["output_dir"] = output_dir
        job["message"] = "Pipeline completed successfully."

    except Exception as e:
        job["status"] = "failed"
        job["message"] = f"Pipeline failed: {str(e)}"
        job["details"] = traceback.format_exc()


# --- Endpoints ---
@app.get("/")
def root():
    """API health check."""
    return {
        "service": "ViralClipper AI",
        "version": "2.0.0",
        "status": "running",
        "endpoints": {
            "generate": "POST /api/v1/generate",
            "status": "GET /api/v1/jobs/{job_id}",
            "cancel": "POST /api/v1/jobs/{job_id}/cancel",
            "download": "GET /api/v1/jobs/{job_id}/download/{filename}",
            "evaluate": "GET /api/v1/jobs/{job_id}/evaluate",
        },
    }


@app.post("/api/v1/generate", response_model=JobStatus)
def generate_clips(request: ClipRequest, background_tasks: BackgroundTasks):
    """Submit a video for clip generation.

    Returns a job_id that can be polled for status updates.
    """
    # Validate video exists
    if not os.path.exists(request.video_path):
        raise HTTPException(status_code=400, detail=f"Video not found: {request.video_path}")

    ext = os.path.splitext(request.video_path)[1].lower()
    if ext not in SUPPORTED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{ext}'. Supported: {SUPPORTED_VIDEO_EXTENSIONS}",
        )

    # Create job
    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "progress": 0,
        "message": "Job queued.",
        "created_at": datetime.now().isoformat(),
        "request": request.model_dump(),
    }

    # Start background task
    background_tasks.add_task(run_pipeline_job, job_id, request)

    return JobStatus(**jobs[job_id])


@app.get("/api/v1/jobs/{job_id}", response_model=JobStatus)
def get_job_status(job_id: str):
    """Check the status of a clip generation job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return JobStatus(**jobs[job_id])


@app.post("/api/v1/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    """Cancel a running job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job = jobs[job_id]
    if job["status"] == "completed":
        return {"message": "Job already completed."}

    job["status"] = "cancelled"
    job["message"] = "Job cancelled by user."
    return {"message": f"Job {job_id} cancelled."}


@app.get("/api/v1/jobs/{job_id}/download/{filename}")
def download_file(job_id: str, filename: str):
    """Download a file from the job's output directory."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job = jobs[job_id]
    if not job.get("output_dir"):
        raise HTTPException(status_code=400, detail="Job output not ready yet.")

    file_path = os.path.join(job["output_dir"], filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")

    return FileResponse(file_path, filename=filename)


@app.get("/api/v1/jobs/{job_id}/evaluate")
def evaluate_job_clips(job_id: str):
    """Run evaluation on the job's output clips."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job = jobs[job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed yet.")
    if not job.get("output_dir"):
        raise HTTPException(status_code=400, detail="No output directory.")

    try:
        from evaluator import evaluate_all_clips, generate_evaluation_report

        output_dir = job["output_dir"]
        clips_json = None
        ass_path = None

        # Find clips.json and subtitles.ass
        for root, dirs, files in os.walk(output_dir):
            for f in files:
                if f == "clips.json":
                    clips_json = os.path.join(root, f)
                if f.endswith(".ass"):
                    ass_path = os.path.join(root, f)

        if not clips_json:
            raise HTTPException(status_code=400, detail="clips.json not found in output.")

        # Find clips directory
        clips_dir = None
        for root, dirs, files in os.walk(output_dir):
            for f in files:
                if f.startswith("clip_") and f.endswith(".mp4"):
                    clips_dir = root
                    break
            if clips_dir:
                break

        if not clips_dir:
            raise HTTPException(status_code=400, detail="No rendered clips found.")

        request_data = job.get("request", {})
        target_duration = request_data.get("target_duration", 90)

        results = evaluate_all_clips(clips_dir, clips_json, ass_path or "", target_duration)
        report_path = generate_evaluation_report(results, os.path.join(output_dir, "EVALUATION.md"))

        # Convert ClipScore objects to dicts for JSON serialization
        serializable = []
        for r in results:
            entry = {k: v for k, v in r.items() if k != "score"}
            if "score" in r:
                score = r["score"]
                entry["score"] = {
                    "overall": score.overall,
                    "audio": score.audio_score,
                    "visual": score.visual_score,
                    "captions": score.caption_score,
                    "engagement": score.engagement_score,
                    "duration": score.duration_score,
                    "details": score.details,
                }
            serializable.append(entry)

        job["evaluation"] = serializable
        return {"results": serializable, "report_path": report_path}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


@app.get("/api/v1/models")
def list_models():
    """List available Ollama models."""
    try:
        import subprocess
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True, text=True,
            creationflags=creationflags,
        )
        models = []
        for line in result.stdout.strip().split("\n")[1:]:
            parts = line.split()
            if parts:
                models.append(parts[0])
        return {"models": models}
    except Exception:
        return {"models": [], "error": "Ollama not available"}


# --- Entry point ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
