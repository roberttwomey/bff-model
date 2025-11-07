#!/usr/bin/env python3
"""Voice chat assistant using Whisper STT, Ollama Gemma 3 Nano, and Piper TTS.

This script performs continuous voice activity detection (VAD) on microphone
audio, automatically segments speech, transcribes each utterance with Whisper,
sends the resulting text to an Ollama model (`gemma3n:e2b` by default), and
plays back the assistant response via Piper text-to-speech.

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
    "you are SNAPPER a robot dog. you do not say woof, whir, tail wag. answer in 2 sentences or less."
)

DEFAULT_OLLAMA_MODEL = os.environ.get("BFF_OLLAMA_MODEL", "gemma3n:e4b")
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
    piper_speed: float = 1.0
    activation_threshold: float = 0.03
    silence_threshold: float = 0.015
    silence_duration: float = 0.8
    min_phrase_seconds: float = 0.5
    block_duration: float = 0.2
    show_levels: bool = False


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
        "--piper-speed",
        type=float,
        default=1.0,
        help="Length scale multiplier for Piper speech rate (lower=faster, default: %(default)s)",
    )
    parser.add_argument(
        "--activation-threshold",
        type=float,
        default=0.03,
        help="RMS amplitude that starts a speech segment (default: %(default)s)",
    )
    parser.add_argument(
        "--silence-threshold",
        type=float,
        default=0.015,
        help="RMS amplitude below which audio counts as silence (default: %(default)s)",
    )
    parser.add_argument(
        "--silence-duration",
        type=float,
        default=0.8,
        help="Seconds of silence that end a speech segment (default: %(default)s)",
    )
    parser.add_argument(
        "--min-phrase-seconds",
        type=float,
        default=0.5,
        help="Discard segments shorter than this many seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--block-duration",
        type=float,
        default=0.2,
        help="Processing block size in seconds for VAD (default: %(default)s)",
    )
    parser.add_argument(
        "--show-levels",
        action="store_true",
        help="Print live RMS level meter to stderr",
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
        piper_speed=args.piper_speed,
        activation_threshold=args.activation_threshold,
        silence_threshold=args.silence_threshold,
        silence_duration=args.silence_duration,
        min_phrase_seconds=args.min_phrase_seconds,
        block_duration=args.block_duration,
        show_levels=args.show_levels,
    )


def load_whisper_model(name: str) -> whisper.Whisper:
    print(f"Loading Whisper model '{name}'…", file=sys.stderr)
    return whisper.load_model(name)


def rms_amplitude(block: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(block))))


def phrase_stream(config: ConversationConfig) -> Iterable[np.ndarray]:
    """Yield successive speech segments detected from the microphone."""

    channels = 1
    block_size = max(1, int(config.sample_rate * config.block_duration))
    silence_blocks_required = max(1, int(config.silence_duration / config.block_duration))
    max_blocks = max(1, int(config.max_record_seconds / config.block_duration))
    min_blocks = max(1, int(config.min_phrase_seconds / config.block_duration))

    q: queue.Queue[np.ndarray] = queue.Queue()

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"[vad] {status}", file=sys.stderr)
        q.put(indata.copy())

    print("Listening continuously… (Ctrl+C to exit)")
    with sd.InputStream(
        samplerate=config.sample_rate,
        channels=channels,
        dtype="float32",
        blocksize=block_size,
        callback=audio_callback,
    ):
        recording = False
        silence_blocks = 0
        collected: List[np.ndarray] = []
        block_counter = 0

        while True:
            block = q.get()
            block_counter += 1 if recording else 0
            amp = rms_amplitude(block)

            if config.show_levels:
                meter_width = 40
                normalized = min(1.0, amp / max(config.activation_threshold, 1e-6))
                filled = int(normalized * meter_width)
                bar = "#" * filled + "-" * (meter_width - filled)
                sys.stderr.write(
                    f"\rLevel {amp:0.3f} |{bar}| {'REC' if recording else '...'}"
                )
                sys.stderr.flush()

            if not recording:
                if amp >= config.activation_threshold:
                    recording = True
                    collected = [block]
                    silence_blocks = 0
                    block_counter = 1
                    print("Speech detected.")
                    if config.show_levels:
                        sys.stderr.write("\n")
                        sys.stderr.flush()
            else:
                collected.append(block)
                if amp < config.silence_threshold:
                    silence_blocks += 1
                else:
                    silence_blocks = 0

                if silence_blocks >= silence_blocks_required or block_counter >= max_blocks:
                    duration = len(collected) * config.block_duration
                    recording = False
                    silence_blocks = 0
                    block_counter = 0

                    if len(collected) < min_blocks:
                        print("Discarded short segment.", file=sys.stderr)
                        collected = []
                        if config.show_levels:
                            sys.stderr.write("\n")
                            sys.stderr.flush()
                        continue

                    audio = np.concatenate(collected, axis=0)
                    collected = []
                    if config.show_levels:
                        sys.stderr.write("\n")
                        sys.stderr.flush()
                    yield audio



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


def synthesize_with_piper(
    voice_path: Path, text: str, output_wav: Path, length_scale: float
) -> None:
    print("Synthesizing speech with Piper…", file=sys.stderr)
    cmd = [
        "piper",
        "--model",
        str(voice_path),
        "--output_file",
        str(output_wav),
    ]
    if length_scale != 1.0:
        cmd.extend(["--length_scale", f"{length_scale:.3f}"])
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

    with tempfile.TemporaryDirectory(prefix="bff-voice-chat-") as tmpdir:
        tmpdir_path = Path(tmpdir)
        try:
            turn = 1
            for phrase in phrase_stream(config):
                raw_audio = tmpdir_path / f"turn-{turn:03d}-input.wav"
                sf.write(raw_audio, phrase, config.sample_rate)

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
                synthesize_with_piper(
                    config.piper_voice,
                    assistant_text,
                    response_audio,
                    length_scale=config.piper_speed,
                )
                play_audio(response_audio)

                turn += 1
        except KeyboardInterrupt:
            print("\nExiting conversation.")


def main() -> None:
    config = parse_args()
    run_conversation(config)


if __name__ == "__main__":
    main()

