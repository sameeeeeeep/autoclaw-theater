"""
Autoclaw Theater — TTS Sidecar for Autoclaw Theater Mode

Pocket TTS (~100M params) voice server with character voices for TV show themes.
CPU-only, ~1x real-time. Checks phrase cache first for instant playback.

Usage:
    autoclaw-theater [--port 7893] [--engine pocket]
    # or: python -m autoclaw_theater.server [--port 7893]

Endpoints:
    GET  /health                → 200 OK
    POST /synthesize            → WAV audio (single line)
    POST /synthesize_dialogue   → {"audio": ["base64wav", ...]} (multi-turn)
    GET  /voices                → list of available voices
"""

import argparse
import asyncio
import base64
import io
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel

app = FastAPI(title="Autoclaw Theater TTS")

# Voice Lab UI
from autoclaw_theater.voicelab import router as voicelab_router
app.include_router(voicelab_router)

# ── Globals ──────────────────────────────────────────────

_engine_name: str = "pocket"  # "pocket" or "kokoro"
_executor = ThreadPoolExecutor(max_workers=2)

# --- Pocket TTS globals ---
_pocket_model = None
_pocket_voices: dict = {}  # voice_id → voice_state

# --- Kokoro globals ---
_kokoro_model = None
_kokoro_cloned: set = set()
_kokoro_cloned_paths: dict = {}

# Voice → reference WAV for Pocket TTS cloning (bundled inside this package)
VOICES_DIR = Path(__file__).parent / "voices"

# Pocket TTS: map character → (ref_wav_filename, builtin_fallback)
POCKET_VOICE_MAP = {
    "gilfoyle":   ("David.wav",  "marius"),
    "dinesh":     ("Moira.wav",  "alba"),
    "david-rose": ("David.wav",  "marius"),
    "moira-rose": ("Moira.wav",  "alba"),
    "rick":       (None,         "jean"),
    "morty":    (None,         "cosette"),
    "sherlock": (None,         "javert"),
    "watson":   (None,         "fantine"),
    "chandler": (None,         "marius"),
    "joey":     (None,         "jean"),
    "dwight":   (None,         "javert"),
    "jim":      (None,         "alba"),
    "jesse":    (None,         "alba"),
    "walter":   (None,         "marius"),
    "tony":     (None,         "jean"),
    "jarvis":   (None,         "javert"),
    "default":  (None,         "alba"),
}

# Kokoro: map character → (cloned_name, preset_fallback)
KOKORO_VOICE_MAP = {
    "gilfoyle": ("gilfoyle", "am_fenrir"),
    "dinesh":   ("dinesh",   "am_michael"),
    "rick":     ("rick",     "am_adam"),
    "morty":    ("morty",    "bf_emma"),
    "sherlock": ("sherlock", "bm_george"),
    "watson":   ("watson",   "bm_lewis"),
    "chandler": ("chandler", "am_michael"),
    "joey":     ("joey",     "am_adam"),
    "dwight":   ("dwight",   "am_fenrir"),
    "jim":      ("jim",      "am_michael"),
    "jesse":    ("jesse",    "am_michael"),
    "walter":   ("walter",   "am_fenrir"),
    "tony":     ("tony",     "am_adam"),
    "jarvis":   ("jarvis",   "bm_george"),
    "default":  ("default",  "af_heart"),
}


# ── Request Models ───────────────────────────────────────

class SynthesizeRequest(BaseModel):
    text: str
    voice_id: str = "default"
    speed: float = 1.0

class DialogueTurn(BaseModel):
    text: str
    voice_id: str = "default"

class DialogueRequest(BaseModel):
    turns: list[DialogueTurn]
    speed: float = 1.0


# ── Phrase Cache ─────────────────────────────────────────

_phrase_cache: dict = {}  # voice_name → { normalized_text → Path }

def _load_phrase_caches():
    cache_dir = Path(__file__).parent / "phrase_cache"
    if not cache_dir.exists():
        return
    for voice_dir in cache_dir.iterdir():
        if not voice_dir.is_dir():
            continue
        manifest_path = voice_dir / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        voice_name = voice_dir.name
        _phrase_cache[voice_name] = {}
        count = 0
        for phrase_id, info in manifest.items():
            wav_path = voice_dir / f"{phrase_id}.wav"
            if wav_path.exists():
                _phrase_cache[voice_name][info["text"].lower().strip().rstrip(".")] = wav_path
                count += 1
        if count:
            print(f"[TTS] Phrase cache: {count} phrases for {voice_name}")

def _check_phrase_cache(text: str, voice_id: str) -> Optional[bytes]:
    voice_name = voice_id.lower()
    cache = _phrase_cache.get(voice_name, {})
    normalized = text.lower().strip().rstrip(".")
    if normalized in cache:
        return cache[normalized].read_bytes()
    return None


# ── Pocket TTS Engine ────────────────────────────────────

def _pocket_init():
    """Load Pocket TTS model and prepare voices."""
    global _pocket_model
    from pocket_tts import TTSModel

    print("[TTS] Loading Pocket TTS...")
    t0 = time.time()
    _pocket_model = TTSModel.load_model()
    print(f"[TTS] Pocket TTS loaded in {time.time() - t0:.1f}s")

    # Pre-load voice states
    _pocket_load_voices()

    # Warmup
    print("[TTS] Warming up Pocket TTS...")
    t0 = time.time()
    if _pocket_voices:
        first_voice = next(iter(_pocket_voices.values()))
        _pocket_model.generate_audio(first_voice, "Hello.")
    print(f"[TTS] Warm-up done in {time.time() - t0:.1f}s")


def _find_voice_wav(filename: str) -> Optional[Path]:
    """Search for a voice WAV file in Resources/Voices/ and its theme subdirectories."""
    # Try top-level first
    top = VOICES_DIR / filename
    if top.exists():
        return top
    # Search theme subdirectories
    if VOICES_DIR.exists():
        for subdir in VOICES_DIR.iterdir():
            if subdir.is_dir() and not subdir.name.startswith('.'):
                candidate = subdir / filename
                if candidate.exists():
                    return candidate
    return None


def _pocket_load_voices():
    """Load or prepare voice states for all characters."""
    global _pocket_voices
    export_dir = Path(__file__).parent / "pocket_voices"
    export_dir.mkdir(exist_ok=True)

    for voice_id, (ref_wav, builtin) in POCKET_VOICE_MAP.items():
        exported = export_dir / f"{voice_id}.safetensors"

        # 1. Check for pre-exported voice state (instant load)
        if exported.exists():
            try:
                state = _pocket_model.get_state_for_audio_prompt(str(exported))
                _pocket_voices[voice_id] = state
                print(f"[TTS] Pocket voice loaded: {voice_id} (exported)")
                continue
            except Exception as e:
                print(f"[TTS] Failed to load exported {voice_id}: {e}")

        # 2. Try cloning from reference WAV (search top-level + theme subdirs)
        if ref_wav:
            wav_path = _find_voice_wav(ref_wav)
            if wav_path and wav_path.exists():
                try:
                    print(f"[TTS] Cloning voice: {voice_id} from {ref_wav}...")
                    t0 = time.time()
                    state = _pocket_model.get_state_for_audio_prompt(str(wav_path))
                    _pocket_voices[voice_id] = state
                    # Export for instant loading next time
                    try:
                        from pocket_tts import export_model_state
                        export_model_state(state, str(exported))
                        print(f"[TTS] Exported {voice_id} → {exported}")
                    except Exception:
                        pass  # export is optional
                    print(f"[TTS] Voice {voice_id} cloned in {time.time() - t0:.1f}s")
                    continue
                except ValueError as e:
                    if "voice cloning" in str(e).lower():
                        print(f"[TTS] Voice cloning not available (HF auth needed), using built-in: {builtin}")
                    else:
                        print(f"[TTS] Clone failed for {voice_id}: {e}")
                except Exception as e:
                    print(f"[TTS] Clone failed for {voice_id}: {e}")

        # 3. Fall back to built-in voice
        try:
            state = _pocket_model.get_state_for_audio_prompt(builtin)
            _pocket_voices[voice_id] = state
            print(f"[TTS] Pocket voice: {voice_id} → built-in '{builtin}'")
        except Exception as e:
            print(f"[TTS] Could not load any voice for {voice_id}: {e}")


def _pocket_synthesize(text: str, voice_id: str, speed: float = 1.0) -> Optional[bytes]:
    """Synthesize with Pocket TTS."""
    if _pocket_model is None:
        return None

    key = voice_id.lower()
    voice_state = _pocket_voices.get(key) or _pocket_voices.get("default")
    if voice_state is None:
        return None

    try:
        t0 = time.time()
        audio = _pocket_model.generate_audio(voice_state, text)
        duration = len(audio) / _pocket_model.sample_rate
        elapsed = time.time() - t0

        # Convert to WAV bytes
        import scipy.io.wavfile
        buf = io.BytesIO()
        scipy.io.wavfile.write(buf, _pocket_model.sample_rate, audio.numpy())
        wav_bytes = buf.getvalue()

        cloned = "★" if POCKET_VOICE_MAP.get(key, (None,))[0] else "○"
        print(f"[TTS] {cloned} pocket [{key}] '{text[:40]}' → {duration:.1f}s in {elapsed:.2f}s ({duration/elapsed:.1f}x RT)")
        return wav_bytes

    except Exception as e:
        print(f"[TTS] Pocket synthesis error: {e}", file=sys.stderr)
        import traceback; traceback.print_exc()
        return None


# ── Kokoro Engine ────────────────────────────────────────

def _kokoro_init(model_name: str = "mlx-community/Kokoro-82M-bf16"):
    """Load Kokoro model and check for cloned voices."""
    global _kokoro_model
    from mlx_audio.utils import load_model

    print(f"[TTS] Loading Kokoro ({model_name})...")
    t0 = time.time()
    _kokoro_model = load_model(model_name)
    print(f"[TTS] Kokoro loaded in {time.time() - t0:.1f}s")

    _kokoro_check_cloned()

    print("[TTS] Warming up Kokoro...")
    t0 = time.time()
    _kokoro_synthesize("Hello.", "gilfoyle", 1.0)
    print(f"[TTS] Warm-up done in {time.time() - t0:.1f}s")


def _kokoro_check_cloned():
    from huggingface_hub import snapshot_download
    try:
        model_path = Path(snapshot_download("mlx-community/Kokoro-82M-bf16", local_files_only=True))
        voices_dir = model_path / "voices"
        for name, (cloned, _) in KOKORO_VOICE_MAP.items():
            sf_path = voices_dir / f"{cloned}.safetensors"
            if sf_path.exists():
                _kokoro_cloned.add(cloned)
                _kokoro_cloned_paths[cloned] = str(sf_path)
                print(f"[TTS] Kokoro cloned voice: {cloned}")
    except Exception as e:
        print(f"[TTS] Kokoro clone check: {e}")


def _kokoro_resolve_voice(voice_id: str) -> str:
    key = voice_id.lower()
    if key in KOKORO_VOICE_MAP:
        cloned, fallback = KOKORO_VOICE_MAP[key]
        if cloned in _kokoro_cloned_paths:
            return _kokoro_cloned_paths[cloned]
        return fallback
    return "af_heart"


def _kokoro_synthesize(text: str, voice_id: str, speed: float = 1.0) -> Optional[bytes]:
    """Synthesize with Kokoro via mlx-audio."""
    if _kokoro_model is None:
        return None
    try:
        t0 = time.time()
        voice = _kokoro_resolve_voice(voice_id)
        audio_chunks = []
        sample_rate = None
        for result in _kokoro_model.generate(text, voice=voice, speed=speed, lang_code="a"):
            audio_chunks.append(result.audio)
            if sample_rate is None:
                sample_rate = result.sample_rate
        if not audio_chunks or sample_rate is None:
            return None
        audio = np.concatenate(audio_chunks)
        from mlx_audio.audio_io import write as audio_write
        buf = io.BytesIO()
        audio_write(buf, audio, sample_rate, format="wav")
        wav_bytes = buf.getvalue()
        duration = len(audio) / sample_rate
        elapsed = time.time() - t0
        print(f"[TTS] ○ kokoro [{voice_id}] '{text[:40]}' → {duration:.1f}s in {elapsed:.2f}s (RTF={elapsed/duration:.2f})")
        return wav_bytes
    except Exception as e:
        print(f"[TTS] Kokoro error: {e}", file=sys.stderr)
        return None


# ── Unified Synthesis ────────────────────────────────────

def _synthesize_one(text: str, voice_id: str, speed: float = 1.0) -> Optional[bytes]:
    """Synthesize text → WAV bytes. Checks phrase cache first, then uses active engine."""
    # 1. Phrase cache (instant)
    cached = _check_phrase_cache(text, voice_id)
    if cached is not None:
        print(f"[TTS] CACHED [{voice_id}] '{text[:40]}'")
        return cached

    # 2. Active engine
    if _engine_name == "pocket":
        return _pocket_synthesize(text, voice_id, speed)
    else:
        return _kokoro_synthesize(text, voice_id, speed)


# ── Endpoints ────────────────────────────────────────────

@app.get("/health")
async def health():
    if _engine_name == "pocket":
        model_loaded = _pocket_model is not None
        voices = sorted(_pocket_voices.keys())
    else:
        model_loaded = _kokoro_model is not None
        voices = sorted(_kokoro_cloned)
    return {
        "status": "ok",
        "engine": _engine_name,
        "model_loaded": model_loaded,
        "voices": voices,
    }


@app.post("/synthesize")
async def synthesize(req: SynthesizeRequest):
    loop = asyncio.get_running_loop()
    wav_bytes = await loop.run_in_executor(
        _executor, _synthesize_one, req.text, req.voice_id, req.speed
    )
    if wav_bytes is None:
        return Response(content=b"synthesis failed", status_code=500)
    return Response(content=wav_bytes, media_type="audio/wav")


@app.post("/synthesize_dialogue")
async def synthesize_dialogue(req: DialogueRequest):
    """Synthesize all turns, return base64-encoded WAVs in order."""
    loop = asyncio.get_running_loop()
    results = []
    for turn in req.turns:
        wav_bytes = await loop.run_in_executor(
            _executor, _synthesize_one, turn.text, turn.voice_id, req.speed
        )
        if wav_bytes:
            results.append(base64.b64encode(wav_bytes).decode("ascii"))
        else:
            results.append(None)
    return {"audio": results}


@app.get("/voices")
async def list_voices():
    if _engine_name == "pocket":
        return {"engine": "pocket", "voices": sorted(_pocket_voices.keys())}
    else:
        return {"engine": "kokoro", "voices": sorted(_kokoro_cloned)}


@app.get("/phrase_cache/stats")
async def phrase_cache_stats():
    stats = {v: len(c) for v, c in _phrase_cache.items()}
    return {"cached_phrases": stats, "total": sum(stats.values())}


# ── Main ─────────────────────────────────────────────────

def main():
    global _engine_name

    parser = argparse.ArgumentParser(description="SiliconValley TTS Sidecar")
    parser.add_argument("--port", type=int, default=7893)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--engine", choices=["pocket", "kokoro"], default="pocket",
                        help="TTS engine: pocket (default, voice cloning) or kokoro (fast presets)")
    args = parser.parse_args()

    _engine_name = args.engine

    # Load phrase caches (shared by both engines)
    _load_phrase_caches()

    # Initialize selected engine
    if _engine_name == "pocket":
        _pocket_init()
    else:
        _kokoro_init()

    print(f"\n[TTS] Server ready on {args.host}:{args.port} (engine: {_engine_name})")

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
