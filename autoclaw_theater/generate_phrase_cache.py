"""
Pre-generate cached voice phrases using Chatterbox (high-quality zero-shot cloning).

These are common reactions, transitions, and filler phrases that characters say frequently.
At runtime, the dialogue generator marks lines as [CACHED:phrase_id] when they match,
and the TTS pipeline serves them instantly from disk instead of generating live.

Usage:
    python generate_phrase_cache.py --voice_name gilfoyle --ref_audio ../Resources/Voices/David.wav
    python generate_phrase_cache.py --all   # generates for all voices with WAVs in Resources/Voices
"""

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

CACHE_DIR = Path(__file__).parent / "phrase_cache"

# Common phrases per character archetype, grouped by category
# The dialogue generator will be prompted to use these exact phrases when appropriate

REACTION_PHRASES = {
    # === Universal reactions (all characters can use) ===
    "_reactions": [
        "Okay I see what's happening here.",
        "Wait wait wait. Hold on.",
        "Oh. Oh no.",
        "That's actually not bad.",
        "Alright, moving on.",
        "Let me take a look at this.",
        "Okay that makes sense.",
        "Interesting. Very interesting.",
        "Well that's a problem.",
        "Alright, so here's the thing.",
        "Right. So basically.",
        "Okay so what we've got here is.",
        "I think I see the issue.",
        "Ha. Classic.",
        "Wait, I think I figured it out.",
        "Nope. That's broken.",
        "Okay here we go.",
        "Not gonna lie, that's pretty clean.",
        "This is fine. Everything is fine.",
        "And there it is. The bug.",
    ],

    # === Gilfoyle-specific ===
    "gilfoyle": [
        "I would rather mass delete my entire codebase than use that approach.",
        "This is what happens when you let a Java developer near a real language.",
        "Dinesh, even your variable names are disappointing.",
        "That's surprisingly not terrible. Don't let it go to your head.",
        "This code is an abomination.",
        "I'm not angry. I'm just disappointed. Actually no, I am angry.",
        "You call that error handling?",
        "This is exactly what I expected. And I expected the worst.",
        "Another day, another Dinesh disaster.",
        "At least the tests pass. That's more than I expected from you.",
        "Dinesh, have you considered a career change?",
        "Oh look. A merge conflict. What a surprise.",
        "The only thing worse than this code is the person who wrote it.",
        "Let me fix this before it embarrasses us further.",
        "I've seen better architecture in a house of cards.",
    ],

    # === Dinesh-specific ===
    "dinesh": [
        "I was literally about to say that exact same thing.",
        "Oh great. Another config file. Just what we needed.",
        "This is fine. Everything is fine. Why wouldn't it be fine?",
        "Wait. Is that a bug or a feature? Asking for a friend.",
        "Okay in my defense, the documentation was terrible.",
        "That's exactly what I was thinking.",
        "I knew that. I totally knew that.",
        "Can we just move on?",
        "Why does everything have to be so complicated?",
        "Okay but my version would have been better.",
        "I don't see what the big deal is.",
        "That's not how I would have done it, but fine.",
        "Oh come on, that's not THAT bad.",
        "Actually, I think the real problem is over here.",
        "See? I told you. Nobody listens to me.",
    ],
}


def generate_cached_phrases(voice_name: str, ref_audio: str, phrases: list[str], overwrite: bool = False):
    """Generate WAV files for a list of phrases using Chatterbox cloning."""
    from mlx_audio.utils import load_model
    from mlx_audio.audio_io import write as audio_write

    voice_cache_dir = CACHE_DIR / voice_name
    voice_cache_dir.mkdir(parents=True, exist_ok=True)

    # Load manifest if exists
    manifest_path = voice_cache_dir / "manifest.json"
    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())

    # Filter phrases that need generating
    to_generate = []
    for phrase in phrases:
        phrase_id = _phrase_id(phrase)
        wav_path = voice_cache_dir / f"{phrase_id}.wav"
        if wav_path.exists() and not overwrite and phrase_id in manifest:
            print(f"  [skip] {phrase[:50]}...")
            continue
        to_generate.append((phrase, phrase_id, wav_path))

    if not to_generate:
        print(f"  All {len(phrases)} phrases already cached for {voice_name}")
        return

    print(f"\n  Loading Chatterbox Turbo for {voice_name}...")
    t0 = time.time()
    model = load_model("mlx-community/chatterbox-turbo-4bit")
    print(f"  Model loaded in {time.time() - t0:.1f}s")

    for i, (phrase, phrase_id, wav_path) in enumerate(to_generate):
        print(f"  [{i+1}/{len(to_generate)}] {phrase[:60]}...")
        t0 = time.time()
        try:
            for result in model.generate(phrase, ref_audio=ref_audio):
                buf = io.BytesIO()
                audio_write(buf, result.audio, result.sample_rate, format="wav")
                wav_path.write_bytes(buf.getvalue())

                duration = len(result.audio) / result.sample_rate
                elapsed = time.time() - t0
                print(f"          {duration:.1f}s audio in {elapsed:.1f}s (RTF={elapsed/duration:.1f})")

                manifest[phrase_id] = {
                    "text": phrase,
                    "duration": round(duration, 2),
                    "generated_at": time.strftime("%Y-%m-%d %H:%M"),
                    "ref_audio": ref_audio,
                }
        except Exception as e:
            print(f"          ERROR: {e}")
            continue

    # Save manifest
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\n  Cached {len(manifest)} phrases for {voice_name}")


def _phrase_id(text: str) -> str:
    """Create a filesystem-safe ID from phrase text."""
    import hashlib
    # Use first 40 chars + hash for uniqueness
    clean = text.lower().replace(" ", "_")[:40]
    clean = "".join(c for c in clean if c.isalnum() or c == "_")
    h = hashlib.md5(text.encode()).hexdigest()[:8]
    return f"{clean}_{h}"


def get_phrases_for_voice(voice_name: str) -> list[str]:
    """Get all phrases that should be cached for a voice."""
    phrases = list(REACTION_PHRASES.get("_reactions", []))
    phrases.extend(REACTION_PHRASES.get(voice_name.lower(), []))
    return phrases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--voice_name", help="Voice name (e.g. gilfoyle)")
    parser.add_argument("--ref_audio", help="Path to reference WAV")
    parser.add_argument("--all", action="store_true", help="Generate for all voices")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing")
    args = parser.parse_args()

    voices_dir = Path(__file__).parent.parent / "Resources" / "Voices"

    if args.all:
        # Process all WAVs in voices dir
        for wav in sorted(voices_dir.glob("*.wav")):
            name = wav.stem.lower()
            phrases = get_phrases_for_voice(name)
            print(f"\n{'='*60}")
            print(f"Voice: {name} ({len(phrases)} phrases)")
            print(f"{'='*60}")
            generate_cached_phrases(name, str(wav), phrases, args.overwrite)
    elif args.voice_name and args.ref_audio:
        phrases = get_phrases_for_voice(args.voice_name)
        print(f"Voice: {args.voice_name} ({len(phrases)} phrases)")
        generate_cached_phrases(args.voice_name, args.ref_audio, phrases, args.overwrite)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
