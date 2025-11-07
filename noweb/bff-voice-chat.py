#!/usr/bin/env python3
"""Voice chat assistant using Whisper STT, Ollama Gemma 3 Nano, and Piper TTS.

This script records speech from the default microphone, transcribes it with
Whisper, sends the resulting text to an Ollama model (`gemma3n:e2b` by default),
and plays back the assistant response via Piper text-to-speech.

Requirements:
    - ollama (Python package) with the `gemma3n:e2b` model pulled locally
    - openai-whisper
    - sounddevice
    - soundfile
    - numpy
    - Piper TTS CLI and at least one voice model file

Example usage:
    python noweb/bff-voice-chat.py --piper-voice piper/en_GB-alan-medium.onnx

Environment variables:
    BFF_OLLAMA_MODEL   override Ollama model name (default: gemma3n:e2b)
    BFF_WHISPER_MODEL  override Whisper model size (default: base)
    BFF_PIPER_VOICE    override Piper voice path if --piper-voice not provided
"""

from __future__ import annotations

import argparse
import os
import queue
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
import ollama
import sounddevice as sd
import soundfile as sf
import whisper


DEFAULT_SYSTEM_PROMPT = (
    "You are BFF, a helpful voice assistant. Keep responses brief and warm."
)

DEFAULT_OLLAMA_MODEL = os.environ.get("BFF_OLLAMA_MODEL", "gemma3n:e2b")
DEFAULT_WHISPER_MODEL = os.environ.get("BFF_WHISPER_MODEL", "base")
DEFAULT_SAMPLE_RATE = 16_000


@dataclass
class ConversationConfig:
    """Runtime configuration for the voice chat assistant."""

    ollama_model: str = DEFAULT_OLLAMA_MODEL
    whisper_model: str = DEFAULT_WHISPER_MODEL
    piper_voice: Path | None = None
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    sample_rate: int = DEFAULT_SAMPLE_RATE
    max_record_seconds: int = 20
    silence_threshold: float = 0.015
    silence_duration: float = 1.0


def parse_args() -> ConversationConfig:
    parser = argparse.ArgumentParser(description="Interactive voice chat assistant")
    parser.add_argument(
        "--ollama-model",
        default=DEFAULT_OLLAMA_MODEL,
        help="Ollama model name to use (default: %(default)s)",
    )
    parser.add_argument(
        "--whisper-model",
        default=DEFAULT_WHISPER_MODEL,
        help="Whisper model size to load (default: %(default)s)",
    )
    parser.add_argument(
        "--piper-voice",
        default=os.environ.get("BFF_PIPER_VOICE"),
        type=Path,
        help="Path to Piper voice model (*.onnx) (default: env BFF_PIPER_VOICE)",
    )
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt sent with each conversation",
    )
    parser.add_argument(
        "--max-record-seconds",
        type=int,
        default=20,
        help="Maximum seconds to record per turn (default: %(default)s)",
    )
    parser.add_argument(
        "--silence-threshold",
        type=float,
        default=0.015,
        help="Amplitude threshold for silence detection (default: %(default)s)",
    )
    parser.add_argument(
        "--silence-duration",
        type=float,
        default=1.0,
        help="Seconds of silence that end recording (default: %(default)s)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help="Audio sample rate for recording and playback (default: %(default)s)",
    )
    args = parser.parse_args()

    if args.piper_voice is None:
        parser.error("Piper voice model must be provided via --piper-voice or BFF_PIPER_VOICE")

    return ConversationConfig(
        ollama_model=args.ollama_model,
        whisper_model=args.whisper_model,
        piper_voice=args.piper_voice,
        system_prompt=args.system_prompt,
        sample_rate=args.sample_rate,
        max_record_seconds=args.max_record_seconds,
        silence_threshold=args.silence_threshold,
        silence_duration=args.silence_duration,
    )


def load_whisper_model(name: str) -> whisper.Whisper:
    print(f"Loading Whisper model '{name}'…", file=sys.stderr)
    return whisper.load_model(name)


def record_until_silence(
    destination: Path,
    sample_rate: int,
    max_seconds: int,
    silence_threshold: float,
    silence_duration: float,
) -> None:
    """Record microphone input to `destination` until silence or timeout."""

    channels = 1
    block_duration = 0.5  # seconds
    block_size = int(sample_rate * block_duration)
    silence_blocks_required = int(max(1, silence_duration / block_duration))
    audio_buffer: List[np.ndarray] = []
    silence_blocks = 0

    q: queue.Queue[np.ndarray] = queue.Queue()

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"[record] {status}", file=sys.stderr)
        q.put(indata.copy())

    print("Listening… speak now (Ctrl+C to cancel)")
    start = time.time()
    with sd.InputStream(
        samplerate=sample_rate,
        channels=channels,
        dtype="float32",
        blocksize=block_size,
        callback=audio_callback,
    ):
        while True:
            try:
                block = q.get(timeout=0.5)
            except queue.Empty:
                block = np.zeros((block_size, channels), dtype="float32")

            audio_buffer.append(block)

            rms = np.sqrt(np.mean(np.square(block)))
            if rms < silence_threshold:
                silence_blocks += 1
            else:
                silence_blocks = 0

            elapsed = time.time() - start
            if silence_blocks >= silence_blocks_required:
                print("Detected silence; stopping recording.")
                break
            if elapsed >= max_seconds:
                print("Reached maximum record time; stopping.")
                break

    audio = np.concatenate(audio_buffer, axis=0)
    sf.write(destination, audio, sample_rate)


def transcribe_audio(model: whisper.Whisper, audio_path: Path) -> str:
    print("Transcribing with Whisper…", file=sys.stderr)
    result = model.transcribe(str(audio_path), fp16=False)
    text = result.get("text", "").strip()
    print(f"You said: {text}")
    return text


def query_ollama(model_name: str, messages: list[dict[str, str]]) -> str:
    print(f"Querying Ollama model '{model_name}'…", file=sys.stderr)
    client = ollama.Client()
    response = client.chat(model=model_name, messages=messages)
    text = response.get("message", {}).get("content", "").strip()
    print(f"Assistant: {text}")
    return text


def synthesize_with_piper(voice_path: Path, text: str, output_wav: Path) -> None:
    print("Synthesizing speech with Piper…", file=sys.stderr)
    cmd = [
        "piper",
        "--model",
        str(voice_path),
        "--output_file",
        str(output_wav),
    ]
    completed = subprocess.run(cmd, input=text, text=True, check=True)
    if completed.returncode != 0:
        raise RuntimeError("Piper synthesis failed")


def play_audio(audio_path: Path) -> None:
    data, samplerate = sf.read(audio_path, dtype="float32")
    sd.play(data, samplerate)
    sd.wait()


def build_initial_messages(system_prompt: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": system_prompt}]


def run_conversation(config: ConversationConfig) -> None:
    whisper_model = load_whisper_model(config.whisper_model)
    messages = build_initial_messages(config.system_prompt)

    print("Voice assistant ready. Press Enter to start recording, or Ctrl+C to exit.")

    with tempfile.TemporaryDirectory(prefix="bff-voice-chat-") as tmpdir:
        tmpdir_path = Path(tmpdir)
        try:
            turn = 1
            while True:
                prompt = input("\nPress Enter to speak (or type 'quit' to exit): ")
                if prompt.strip().lower() in {"q", "quit", "exit"}:
                    print("Goodbye!")
                    break

                raw_audio = tmpdir_path / f"turn-{turn:03d}-input.wav"
                record_until_silence(
                    raw_audio,
                    sample_rate=config.sample_rate,
                    max_seconds=config.max_record_seconds,
                    silence_threshold=config.silence_threshold,
                    silence_duration=config.silence_duration,
                )

                user_text = transcribe_audio(whisper_model, raw_audio)
                if not user_text:
                    print("Did not catch that. Let's try again.")
                    continue

                messages.append({"role": "user", "content": user_text})
                assistant_text = query_ollama(config.ollama_model, messages)
                if not assistant_text:
                    print("Assistant returned empty response; stopping.")
                    break

                messages.append({"role": "assistant", "content": assistant_text})

                response_audio = tmpdir_path / f"turn-{turn:03d}-response.wav"
                synthesize_with_piper(config.piper_voice, assistant_text, response_audio)
                play_audio(response_audio)

                turn += 1
        except KeyboardInterrupt:
            print("\nExiting conversation.")


def main() -> None:
    config = parse_args()
    run_conversation(config)


if __name__ == "__main__":
    main()

