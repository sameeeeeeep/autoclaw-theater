"""
Voice Lab — Web UI for managing voice cloning jobs.

Endpoints added to the main FastAPI app:
  GET  /voicelab              → HTML UI
  GET  /voicelab/status       → JSON status of all voices + active jobs
  POST /voicelab/upload       → Upload WAV + start cloning
  POST /voicelab/reclone      → Re-run cloning for existing voice
  GET  /voicelab/preview/:name → Play current Kokoro output for a voice
  POST /voicelab/stop/:job_id → Stop a running clone job
"""

import asyncio
import base64
import glob
import io
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, Response

router = APIRouter(prefix="/voicelab")

# ── State ───────────────────────────────────────────────

_jobs: dict = {}  # job_id → {name, status, progress, score, pid, log, started_at}
_voices_dir = Path(__file__).parent.parent / "Resources" / "Voices"
_sidecar_dir = Path(__file__).parent


def _get_voice_info():
    """Scan for all reference WAVs and their cloning status."""
    voices = []
    wav_dir = _voices_dir
    if not wav_dir.exists():
        wav_dir.mkdir(parents=True, exist_ok=True)

    for wav in sorted(wav_dir.glob("*.wav")):
        name = wav.stem.lower()
        # Check if cloned tensor exists
        tensor_paths = glob.glob(
            os.path.expanduser(f"~/.cache/huggingface/hub/models--*Kokoro*/snapshots/*/voices/{name}.safetensors")
        )
        has_tensor = len(tensor_paths) > 0
        tensor_size = os.path.getsize(tensor_paths[0]) if has_tensor else 0
        wav_size = wav.stat().st_size
        wav_duration = wav_size / (16000 * 2)  # rough estimate for 16kHz 16-bit

        # Check if there's a running job for this voice
        active_job = None
        for jid, job in _jobs.items():
            if job["name"] == name and job["status"] in ("running", "queued"):
                active_job = jid
                break

        voices.append({
            "name": name,
            "wav_path": str(wav),
            "wav_size_kb": round(wav_size / 1024),
            "wav_duration_s": round(wav_duration, 1),
            "has_tensor": has_tensor,
            "tensor_size_kb": round(tensor_size / 1024) if has_tensor else 0,
            "active_job": active_job,
        })
    return voices


def _read_job_log(job_id: str, last_n: int = 50) -> list[str]:
    """Read last N lines from a job's log file."""
    log_path = f"/tmp/voicelab_{job_id}.log"
    if not os.path.exists(log_path):
        return []
    with open(log_path, "r") as f:
        lines = f.readlines()
    return [l.rstrip() for l in lines[-last_n:]]


def _parse_progress(log_lines: list[str]) -> dict:
    """Parse cloning log to extract progress info."""
    info = {"step": 0, "score": 0.0, "sigma": 0.0, "phase": "starting"}
    for line in reversed(log_lines):
        if "Step" in line and "score=" in line:
            try:
                parts = line.split("score=")[1]
                info["score"] = float(parts.split()[0])
                if "σ=" in line:
                    info["sigma"] = float(line.split("σ=")[1].split()[0])
                step_part = line.split("Step")[1].strip().split(":")[0].strip()
                info["step"] = int(step_part)
                info["phase"] = "optimizing"
            except (ValueError, IndexError):
                pass
            break
        elif "Step 1:" in line or "Generate cloned samples" in line:
            info["phase"] = "generating_samples"
        elif "Extract speaker" in line:
            info["phase"] = "extracting_embedding"
        elif "Find closest" in line:
            info["phase"] = "finding_preset"
        elif "Optimize voice" in line:
            info["phase"] = "optimizing"
        elif "Final score" in line:
            try:
                info["score"] = float(line.split("Final score:")[1].strip())
                info["phase"] = "done"
            except (ValueError, IndexError):
                pass
        elif "Target" in line and "reached" in line:
            info["phase"] = "done"
    return info


# ── Endpoints ───────────────────────────────────────────

@router.get("/", response_class=HTMLResponse)
async def voicelab_ui():
    html_path = Path(__file__).parent / "voicelab.html"
    return html_path.read_text()


@router.get("/status")
async def voicelab_status():
    voices = _get_voice_info()

    # Update job statuses
    for jid, job in list(_jobs.items()):
        if job["status"] == "running" and job.get("pid"):
            # Check if process still alive
            try:
                os.kill(job["pid"], 0)
            except ProcessLookupError:
                job["status"] = "completed"
        log_lines = _read_job_log(jid)
        progress = _parse_progress(log_lines)
        job["progress"] = progress
        job["log_tail"] = log_lines[-10:]

    return {
        "voices": voices,
        "jobs": {jid: {k: v for k, v in j.items() if k != "process"} for jid, j in _jobs.items()},
    }


@router.post("/upload")
async def upload_voice(file: UploadFile = File(...), name: str = Form(...)):
    """Upload a WAV file and optionally start cloning."""
    name = name.lower().strip().replace(" ", "_")
    if not name:
        return JSONResponse({"error": "Name required"}, status_code=400)

    # Save WAV
    wav_path = _voices_dir / f"{name}.wav"
    _voices_dir.mkdir(parents=True, exist_ok=True)
    content = await file.read()
    with open(wav_path, "wb") as f:
        f.write(content)

    # Auto-start cloning
    job_id = _start_clone_job(name, str(wav_path))
    return {"ok": True, "name": name, "job_id": job_id, "wav_path": str(wav_path)}


@router.post("/reclone")
async def reclone_voice(name: str = Form(...), steps: int = Form(2000), target: float = Form(0.90)):
    """Re-run cloning for an existing voice."""
    name = name.lower().strip()
    wav_path = _voices_dir / f"{name}.wav"
    # Also check capitalized
    if not wav_path.exists():
        for f in _voices_dir.glob("*.wav"):
            if f.stem.lower() == name:
                wav_path = f
                break
    if not wav_path.exists():
        return JSONResponse({"error": f"No WAV found for '{name}'"}, status_code=404)

    job_id = _start_clone_job(name, str(wav_path), steps=steps, target=target)
    return {"ok": True, "name": name, "job_id": job_id}


@router.post("/stop/{job_id}")
async def stop_job(job_id: str):
    if job_id not in _jobs:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    job = _jobs[job_id]
    if job.get("pid"):
        try:
            os.kill(job["pid"], 9)
        except ProcessLookupError:
            pass
    job["status"] = "stopped"
    return {"ok": True}


@router.get("/preview/{name}")
async def preview_voice(name: str):
    """Generate a short preview with the current voice tensor."""
    from autoclaw_theater.server import _model, _synthesize_one
    if _model is None:
        return JSONResponse({"error": "Model not loaded"}, status_code=503)

    wav_bytes = _synthesize_one(
        "This is a preview of the cloned voice. How does it sound to you?",
        name, 1.0
    )
    if wav_bytes is None:
        return JSONResponse({"error": "Synthesis failed"}, status_code=500)
    return Response(content=wav_bytes, media_type="audio/wav")


# ── Clone Job Runner ────────────────────────────────────

def _start_clone_job(name: str, wav_path: str, steps: int = 2000, target: float = 0.90) -> str:
    job_id = str(uuid.uuid4())[:8]
    log_path = f"/tmp/voicelab_{job_id}.log"

    venv_python = str(_sidecar_dir / ".venv" / "bin" / "python3")
    clone_script = str(_sidecar_dir / "clone_voice.py")

    cmd = [
        venv_python, clone_script,
        "--ref_audio", wav_path,
        "--voice_name", name,
        "--steps", str(steps),
        "--target", str(target),
        "--samples", "8",
    ]

    with open(log_path, "w") as log_file:
        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=str(_sidecar_dir),
        )

    _jobs[job_id] = {
        "name": name,
        "status": "running",
        "pid": proc.pid,
        "started_at": time.time(),
        "steps": steps,
        "target": target,
        "wav_path": wav_path,
        "progress": {"step": 0, "score": 0.0, "phase": "starting"},
        "log_tail": [],
    }
    return job_id
