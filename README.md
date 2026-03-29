<p align="center">
  <strong>Autoclaw Theater</strong>
</p>

<h1 align="center">autoclaw-theater</h1>

<p align="center">
  TTS voice sidecar for <a href="https://github.com/sameeeeeeep/autoclaw">Autoclaw</a> Theater Mode — Pocket TTS voice server with character voices.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/engine-Pocket%20TTS-blue" />
  <img src="https://img.shields.io/badge/python-3.10%2B-green" />
  <img src="https://img.shields.io/badge/license-MIT-green" />
</p>

---

## What is this?

Autoclaw Theater Mode shows an animated Picture-in-Picture window where TV characters explain your coding session ELI5-style. This package is the **optional** TTS voice server that gives those characters actual voices.

It runs a lightweight FastAPI server on port 7893, using [Pocket TTS](https://github.com/sweetcocoa/pocket-tts) (~100M params) for voice synthesis. CPU-only, no GPU required, ~1x real-time.

**You don't need this for Theater Mode to work** — Autoclaw falls back to text-only dialog if the sidecar isn't installed.

## 8 Character Pairs

| Theme | Characters | Voice Style |
|-------|-----------|-------------|
| Silicon Valley | Gilfoyle & Dinesh | Cloned from reference WAVs |
| Schitt's Creek | David & Moira | Cloned from reference WAVs |
| The Office | Dwight & Jim | Built-in Pocket TTS voices |
| Friends | Chandler & Joey | Built-in Pocket TTS voices |
| Rick and Morty | Rick & Morty | Built-in Pocket TTS voices |
| Sherlock | Sherlock & Watson | Built-in Pocket TTS voices |
| Breaking Bad | Jesse & Walter | Built-in Pocket TTS voices |
| Iron Man | Tony & JARVIS | Built-in Pocket TTS voices |

## Install

```bash
pip install autoclaw-theater
```

Or install from source:

```bash
git clone https://github.com/sameeeeeeep/autoclaw-theater.git
cd autoclaw-theater
pip install -e .
```

### Requirements

- Python 3.10+
- ~500MB disk (Pocket TTS model downloads on first run)
- Works on macOS (Apple Silicon + Intel) and Linux

## Usage

### As a CLI

```bash
autoclaw-theater                    # default: port 7893
autoclaw-theater --port 8000        # custom port
```

### As a module

```bash
python -m autoclaw_theater.server --port 7893
```

### With Autoclaw

When Autoclaw detects Theater Mode is enabled, it automatically launches this server and connects to it. No manual setup needed — just install the package and Autoclaw handles the rest.

## API

### `GET /health`
Returns `200 OK` when the server is ready.

### `POST /synthesize`
Synthesize a single line of speech.

```json
{
  "text": "Have you tried turning it off and on again?",
  "voice": "gilfoyle",
  "speed": 1.0
}
```

Returns: `audio/wav`

### `POST /synthesize_dialogue`
Synthesize a multi-turn dialog (used by Autoclaw Theater Mode).

```json
{
  "lines": [
    {"voice": "gilfoyle", "text": "He's trying to center a div again."},
    {"voice": "dinesh", "text": "At least he's not using tables this time."}
  ],
  "speed": 1.0
}
```

Returns:
```json
{
  "audio": ["base64-encoded-wav", "base64-encoded-wav"]
}
```

### `GET /voices`
List all available voices.

## Voice Cloning

The server includes a Voice Lab web UI for cloning new voices:

1. Start the server: `autoclaw-theater`
2. Open `http://localhost:7893/voicelab`
3. Upload a reference WAV (10-30s of clean speech)
4. Clone and preview

Cloned voices are saved as `.safetensors` files in the `pocket_voices/` directory.

## Architecture

```
Autoclaw (Swift)
  │
  ├── Launches: autoclaw-theater --port 7893
  │
  ├── POST /synthesize_dialogue  (batch TTS for dialog lines)
  │
  └── Plays WAV responses sequentially via AVAudioPlayer
```

The sidecar is a standalone process — Autoclaw spawns it on Theater Mode activation and terminates it on quit. If the sidecar isn't installed or fails to start, Theater Mode gracefully falls back to text-only dialog.

## License

[MIT](LICENSE) — The Last Prompt, 2025.
