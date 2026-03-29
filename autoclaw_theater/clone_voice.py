"""
Voice Cloner for Kokoro TTS

Pipeline:
1. Load Chatterbox Turbo to generate cloned speech samples from reference WAV
2. Use Resemblyzer to extract speaker embedding from those samples
3. Optimize a Kokoro voice tensor to match that embedding
4. Save as .safetensors for Kokoro to use at full speed

Usage:
    python clone_voice.py --ref_audio ../Resources/Voices/David.wav --voice_name gilfoyle
"""

import argparse
import io
import os
import sys
import time
from pathlib import Path

import numpy as np


def step1_generate_cloned_samples(ref_wav: str, num_sentences: int = 8):
    """Use Chatterbox Turbo to generate speech in the cloned voice."""
    print("\n=== Step 1: Generate cloned samples via Chatterbox ===")
    from mlx_audio.utils import load_model
    from mlx_audio.audio_io import write as audio_write

    sentences = [
        "The fundamental problem with this architecture is that it was designed by someone who clearly doesn't understand distributed systems.",
        "I've seen better code written by a random number generator. At least that would have some mathematical elegance.",
        "This is exactly the kind of technical debt that will haunt us for the next three years.",
        "The solution is obvious if you've read any paper on consensus algorithms published in the last decade.",
        "Every time I look at this codebase I find a new reason to question the hiring process.",
        "We need to refactor the entire authentication module before it becomes completely unmaintainable.",
        "The performance metrics clearly show that this approach scales about as well as a bicycle on a highway.",
        "I'm not saying it's impossible, I'm saying it would require competence that I haven't seen demonstrated yet.",
    ]

    print(f"Loading Chatterbox Turbo 4-bit...")
    t0 = time.time()
    model = load_model("mlx-community/chatterbox-turbo-4bit")
    print(f"Loaded in {time.time()-t0:.1f}s")

    all_audio = []
    sample_rate = None

    for i, text in enumerate(sentences[:num_sentences]):
        print(f"  Generating [{i+1}/{num_sentences}]: {text[:50]}...")
        t0 = time.time()
        for result in model.generate(text, ref_audio=ref_wav):
            all_audio.append(result.audio)
            if sample_rate is None:
                sample_rate = result.sample_rate
        print(f"    Done in {time.time()-t0:.1f}s")

    # Concatenate all audio
    combined = np.concatenate(all_audio)
    duration = len(combined) / sample_rate
    print(f"Total cloned audio: {duration:.1f}s at {sample_rate}Hz")

    # Save combined audio
    buf = io.BytesIO()
    audio_write(buf, combined, sample_rate, format="wav")
    return buf.getvalue(), sample_rate


def step2_extract_speaker_embedding(cloned_wav_bytes: bytes):
    """Extract speaker embedding from cloned audio using Resemblyzer."""
    print("\n=== Step 2: Extract speaker embedding ===")
    from resemblyzer import VoiceEncoder, preprocess_wav
    import soundfile as sf

    encoder = VoiceEncoder()

    # Load audio from bytes
    buf = io.BytesIO(cloned_wav_bytes)
    audio, sr = sf.read(buf)
    wav = preprocess_wav(audio, source_sr=sr)
    embedding = encoder.embed_utterance(wav)
    print(f"Speaker embedding shape: {embedding.shape}")
    return embedding, encoder


def step3_find_closest_preset(target_embedding, encoder):
    """Find which Kokoro preset voice is closest to the target."""
    print("\n=== Step 3: Find closest Kokoro preset ===")
    from resemblyzer import preprocess_wav
    from huggingface_hub import snapshot_download
    from mlx_audio.utils import load_model
    from mlx_audio.audio_io import write as audio_write
    import soundfile as sf

    model = load_model("mlx-community/Kokoro-82M-bf16")

    # Warmup
    for r in model.generate("warmup", voice="am_adam", speed=1.0, lang_code="a"):
        pass

    test_text = "The fundamental problem with this architecture is obvious."
    presets = ["am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam",
               "am_michael", "am_onyx", "am_puck",
               "bm_daniel", "bm_fable", "bm_george", "bm_lewis"]

    best_score = -1
    best_preset = None
    scores = {}

    for preset in presets:
        # Generate audio with this preset
        for r in model.generate(test_text, voice=preset, speed=1.0, lang_code="a"):
            buf = io.BytesIO()
            audio_write(buf, r.audio, r.sample_rate, format="wav")
            buf.seek(0)
            audio, sr = sf.read(buf)
            wav = preprocess_wav(audio, source_sr=sr)
            emb = encoder.embed_utterance(wav)
            score = float(np.dot(target_embedding, emb))
            scores[preset] = score
            if score > best_score:
                best_score = score
                best_preset = preset

    # Print ranked
    for preset, score in sorted(scores.items(), key=lambda x: -x[1]):
        marker = " <<<" if preset == best_preset else ""
        print(f"  {preset}: {score:.4f}{marker}")

    print(f"\nClosest preset: {best_preset} (similarity: {best_score:.4f})")
    return best_preset, model


def evaluate_tensor(candidate_tensor, kokoro_model, encoder, target_embedding, sentences, voice_name):
    """Generate audio with a candidate tensor and return avg cosine similarity."""
    import mlx.core as mx
    from mlx_audio.audio_io import write as audio_write
    from resemblyzer import preprocess_wav
    import soundfile as sf

    tmp_path = f"/tmp/kokoro_candidate_{voice_name}.safetensors"
    mx.save_safetensors(tmp_path, {"voice": mx.array(candidate_tensor)})

    scores = []
    for text in sentences:
        try:
            for r in kokoro_model.generate(text, voice=tmp_path, speed=1.0, lang_code="a"):
                buf = io.BytesIO()
                audio_write(buf, r.audio, r.sample_rate, format="wav")
                buf.seek(0)
                audio, sr = sf.read(buf)
                wav = preprocess_wav(audio, source_sr=sr)
                emb = encoder.embed_utterance(wav)
                scores.append(float(np.dot(target_embedding, emb)))
        except Exception:
            continue
    return np.mean(scores) if scores else 0.0


def step4_optimize_voice_tensor(
    target_embedding, encoder, best_preset, kokoro_model,
    steps=2000, voice_name="custom", target_score=0.90,
    population_size=8, sigma_init=0.03
):
    """Optimize a Kokoro voice tensor using CMA-ES-lite (mu+lambda) evolution."""
    print(f"\n=== Step 4: Optimize voice tensor (up to {steps} steps, target={target_score}) ===")
    import mlx.core as mx
    from huggingface_hub import snapshot_download

    # Load the best preset tensor as starting point
    model_path = snapshot_download("mlx-community/Kokoro-82M-bf16", local_files_only=True)
    weights = mx.load(f"{model_path}/voices/{best_preset}.safetensors")
    mean = np.array(weights["voice"], copy=True).astype(np.float32).flatten()
    dim = mean.shape[0]
    sigma = sigma_init

    # Diverse test sentences for robust evaluation
    eval_sentences = [
        "The fundamental problem with this architecture is obvious to anyone who understands it.",
        "This approach will not scale beyond a trivial workload under real conditions.",
        "Every time I look at this code I find a new reason to question everything.",
    ]
    # Quick eval uses 1 sentence for speed, full eval uses all 3
    quick_sentences = eval_sentences[:1]

    # Evaluate starting point
    best_score = evaluate_tensor(mean, kokoro_model, encoder, target_embedding, eval_sentences, voice_name)
    best_mean = mean.copy()
    print(f"  Baseline ({best_preset}): {best_score:.4f}")

    # Also try blending top-2 presets for a better start
    # (skip for now, the CMA loop will handle it)

    stagnation = 0
    for step in range(steps):
        # Generate population of candidates
        noise = np.random.randn(population_size, dim).astype(np.float32)
        candidates = mean[None, :] + sigma * noise  # (pop, dim)

        # Evaluate each candidate (quick — 1 sentence)
        scores = []
        for i in range(population_size):
            s = evaluate_tensor(
                candidates[i].reshape(weights["voice"].shape),
                kokoro_model, encoder, target_embedding,
                quick_sentences, voice_name
            )
            scores.append(s)

        scores = np.array(scores)
        ranking = np.argsort(-scores)  # best first

        # Select top half (mu)
        mu = population_size // 2
        top_indices = ranking[:mu]

        # Weighted recombination (rank-based weights)
        log_weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = log_weights / log_weights.sum()

        # Update mean
        old_mean = mean.copy()
        mean = np.zeros(dim, dtype=np.float32)
        for i, idx in enumerate(top_indices):
            mean += w[i] * candidates[idx]

        # Adaptive sigma: increase if improving, decrease if stagnating
        top_score = scores[ranking[0]]
        if top_score > best_score + 0.001:
            # Full evaluation on best candidate
            full_score = evaluate_tensor(
                candidates[ranking[0]].reshape(weights["voice"].shape),
                kokoro_model, encoder, target_embedding,
                eval_sentences, voice_name
            )
            if full_score > best_score:
                best_score = full_score
                best_mean = candidates[ranking[0]].copy()
                stagnation = 0
                print(f"  Step {step:4d}: score={best_score:.4f} σ={sigma:.5f} (pop_best={top_score:.4f})")
                if best_score >= target_score:
                    print(f"  🎯 Target {target_score} reached!")
                    break
            else:
                stagnation += 1
        else:
            stagnation += 1

        # Sigma adaptation
        if stagnation > 10:
            sigma *= 0.85  # shrink search radius
            stagnation = 0
        elif stagnation == 0:
            sigma = min(sigma * 1.05, sigma_init * 3)  # expand slightly on improvement

        # Status every 50 steps
        if step > 0 and step % 50 == 0:
            print(f"  Step {step:4d}: best={best_score:.4f} σ={sigma:.5f} (pop: {scores.mean():.4f}±{scores.std():.4f})")

        # Early stop if sigma collapsed
        if sigma < 1e-5:
            print(f"  Sigma collapsed at step {step}, stopping.")
            break

    print(f"\nFinal score: {best_score:.4f}")

    # Save final voice tensor
    final_tensor = best_mean.reshape(weights["voice"].shape)

    # Save to both Kokoro model paths
    for model_name in ["mlx-community--Kokoro-82M-bf16", "prince-canuma--Kokoro-82M"]:
        import glob
        paths = glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/models--{model_name}/snapshots/*/voices/"))
        for p in paths:
            out = os.path.join(p, f"{voice_name}.safetensors")
            mx.save_safetensors(out, {"voice": mx.array(final_tensor)})
            print(f"Saved: {out}")

    # Backup
    backup = Path(__file__).parent / f"{voice_name}_voice.safetensors"
    mx.save_safetensors(str(backup), {"voice": mx.array(final_tensor)})
    print(f"Backup: {backup}")

    return str(backup)


def main():
    parser = argparse.ArgumentParser(description="Clone a voice for Kokoro TTS")
    parser.add_argument("--ref_audio", required=True, help="Path to reference WAV")
    parser.add_argument("--voice_name", required=True, help="Name for the cloned voice (e.g. gilfoyle)")
    parser.add_argument("--steps", type=int, default=2000, help="Optimization steps")
    parser.add_argument("--target", type=float, default=0.90, help="Target similarity score")
    parser.add_argument("--samples", type=int, default=6, help="Chatterbox samples to generate")
    args = parser.parse_args()

    t_start = time.time()

    # Step 1: Generate cloned audio via Chatterbox
    cloned_wav, sr = step1_generate_cloned_samples(args.ref_audio, args.samples)

    # Save intermediate cloned audio
    with open(f"/tmp/cloned_{args.voice_name}.wav", "wb") as f:
        f.write(cloned_wav)
    print(f"Saved cloned reference to /tmp/cloned_{args.voice_name}.wav")

    # Step 2: Extract speaker embedding
    target_emb, encoder = step2_extract_speaker_embedding(cloned_wav)

    # Also extract from original reference for comparison
    from resemblyzer import preprocess_wav
    import soundfile as sf
    orig_audio, orig_sr = sf.read(args.ref_audio)
    orig_wav = preprocess_wav(orig_audio, source_sr=orig_sr)
    orig_emb = encoder.embed_utterance(orig_wav)
    print(f"Original → Cloned similarity: {float(np.dot(orig_emb, target_emb)):.4f}")

    # Step 3: Find closest preset
    best_preset, kokoro_model = step3_find_closest_preset(target_emb, encoder)

    # Step 4: Optimize with CMA-ES
    output_path = step4_optimize_voice_tensor(
        target_emb, encoder, best_preset, kokoro_model,
        steps=args.steps, voice_name=args.voice_name,
        target_score=args.target
    )

    total = time.time() - t_start
    print(f"\n{'='*50}")
    print(f"Done! Total time: {total/60:.1f} minutes")
    print(f"Voice saved to: {output_path}")
    print(f"Use in Kokoro: voice='{args.voice_name}'")


if __name__ == "__main__":
    main()
