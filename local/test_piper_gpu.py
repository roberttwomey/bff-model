#!/usr/bin/env python3
"""Quick Piper TTS sanity check that reports whether CUDA is in use."""

from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

from piper import PiperVoice

DEFAULT_TEXT = "This is a quick Piper test. If you hear me, synthesis worked."


def default_voice_path() -> Path:
    """Return the bundled voice path if present."""
    root = Path(__file__).resolve().parent
    candidate = root / "piper" / "en_GB-alan-medium.onnx"
    return candidate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--voice",
        type=Path,
        default=default_voice_path(),
        help="Path to a Piper voice .onnx file (defaults to en_GB-alan-medium).",
    )
    parser.add_argument(
        "--text",
        default=DEFAULT_TEXT,
        help="Text to synthesize.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("piper-test.wav"),
        help="Destination WAV path (default: ./piper-test.wav).",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU execution (useful for comparison).",
    )
    return parser.parse_args()


def summarize_providers(voice: PiperVoice) -> tuple[list[str], str]:
    providers = voice.session.get_providers()
    active = providers[0] if providers else "unknown"
    return providers, active


def synthesize_to_wav(voice: PiperVoice, text: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as wav_file:
        voice.synthesize_wav(text, wav_file)


def main() -> None:
    args = parse_args()
    voice_path = args.voice.expanduser()
    if not voice_path.exists():
        raise SystemExit(f"Piper voice not found: {voice_path}")

    use_cuda = not args.cpu
    print(f"Loading {voice_path} with {'CUDA' if use_cuda else 'CPU'} backend…")
    voice = PiperVoice.load(
        str(voice_path),
        use_cuda=use_cuda,
    )

    providers, active = summarize_providers(voice)
    print(f"Available providers: {providers}")
    print(f"Active provider: {active}")
    if use_cuda and "CUDAExecutionProvider" not in providers:
        print(
            "Warning: CUDA provider unavailable. Check onnxruntime-gpu installation.",
            file=sys.stderr,
        )
    elif use_cuda and active != "CUDAExecutionProvider":
        print(
            "Warning: CUDA provider is present but not active.",
            file=sys.stderr,
        )

    synthesize_to_wav(voice, args.text, args.output.resolve())
    print(f"Wrote synthesized audio to {args.output.resolve()}")


if __name__ == "__main__":
    main()

