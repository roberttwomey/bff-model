#!/usr/bin/env python3
"""Voice chat assistant using Whisper STT, Ollama Gemma 3 Nano, and Piper TTS.

This script performs continuous voice activity detection (VAD) on microphone
audio, automatically segments speech, transcribes each utterance with Whisper,
sends the resulting text to an Ollama model (`gemma3n:e2b` by default), and
plays back the assistant response via Piper text-to-speech using the Python
`piper-tts` library.

Requirements:
    - ollama (Python package) with the `gemma3n:e2b` model pulled locally
    - openai-whisper
    - sounddevice
    - soundfile
    - numpy
    - piper-tts (Python package) and at least one Piper voice model file

Example usage:
    python noweb/bff-voice-chat.py --piper-voice piper/en_GB-alan-medium.onnx

Environment variables:
    BFF_OLLAMA_MODEL   override Ollama model name (default: gemma3n:e2b)
    BFF_WHISPER_MODEL  override Whisper model size (default: base)
    BFF_PIPER_VOICE    override Piper voice path if --piper-voice not provided
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, List
import wave

import numpy as np
import ollama
import sounddevice as sd
import soundfile as sf
import whisper

from piper import PiperVoice
try:  # Optional type that some versions expose
    from piper import AudioChunk  # type: ignore
except ImportError:  # pragma: no cover - older library versions
    AudioChunk = None


DEFAULT_SYSTEM_PROMPT = (
    "you are SNAPPER a robot dog. you do not say woof, whir, tail wag. answer in 2 sentences or less."
)

DEFAULT_OLLAMA_MODEL = os.environ.get("BFF_OLLAMA_MODEL", "gemma3n:e4b")
DEFAULT_WHISPER_MODEL = os.environ.get("BFF_WHISPER_MODEL", "base")
DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_INPUT_DEVICE_KEYWORD = os.environ.get(
    "BFF_INPUT_DEVICE_KEYWORD", "OpenRun Pro 2 by Shokz"
)
LOG_ROOT = Path.home() / "bff" / "logs"


@dataclass
class ConversationConfig:
    """Runtime configuration for the voice chat assistant."""

    ollama_model: str = DEFAULT_OLLAMA_MODEL
    whisper_model: str = DEFAULT_WHISPER_MODEL
    piper_voice: Path | None = None
    piper_config: Path | None = None
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    sample_rate: int = DEFAULT_SAMPLE_RATE
    max_record_seconds: int = 20
    piper_length_scale: float | None = None
    piper_noise_scale: float | None = None
    piper_noise_w: float | None = None
    activation_threshold: float = 0.03
    silence_threshold: float = 0.015
    silence_duration: float = 0.8
    min_phrase_seconds: float = 0.5
    block_duration: float = 0.2
    show_levels: bool = False
    input_device_keyword: str | None = DEFAULT_INPUT_DEVICE_KEYWORD
    input_device_index: int | None = None


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
        "--piper-config",
        type=Path,
        help="Optional path to Piper voice config (*.json); defaults to <voice>.json",
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
        "--piper-length-scale",
        type=float,
        help="Override Piper config length_scale (lower=faster)",
    )
    parser.add_argument(
        "--piper-noise-scale",
        type=float,
        help="Override Piper config noise_scale",
    )
    parser.add_argument(
        "--piper-noise-w",
        type=float,
        help="Override Piper config noise_w",
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
        "--input-device-keyword",
        default=DEFAULT_INPUT_DEVICE_KEYWORD,
        help="Substring to match desired input device (default: %(default)s)",
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
    if not args.piper_voice.exists():
        parser.error(f"Piper voice model not found: {args.piper_voice}")

    input_keyword = args.input_device_keyword.strip() if args.input_device_keyword else None
    if input_keyword == "":
        input_keyword = None

    return ConversationConfig(
        ollama_model=args.ollama_model,
        whisper_model=args.whisper_model,
        piper_voice=args.piper_voice,
        piper_config=args.piper_config,
        system_prompt=args.system_prompt,
        sample_rate=args.sample_rate,
        max_record_seconds=args.max_record_seconds,
        piper_length_scale=args.piper_length_scale,
        piper_noise_scale=args.piper_noise_scale,
        piper_noise_w=args.piper_noise_w,
        activation_threshold=args.activation_threshold,
        silence_threshold=args.silence_threshold,
        silence_duration=args.silence_duration,
        min_phrase_seconds=args.min_phrase_seconds,
        block_duration=args.block_duration,
        show_levels=args.show_levels,
        input_device_keyword=input_keyword,
    )


def load_whisper_model(name: str) -> whisper.Whisper:
    print(f"Loading Whisper model '{name}'…", file=sys.stderr)
    return whisper.load_model(name)


def resolve_piper_config_path(model_path: Path, config_path: Path | None) -> Path:
    if config_path is not None:
        if not config_path.exists():
            raise FileNotFoundError(f"Piper config not found: {config_path}")
        return config_path

    candidate = model_path.with_suffix(model_path.suffix + ".json")
    if candidate.exists():
        return candidate

    alt = model_path.with_suffix(".json")
    if alt.exists():
        return alt

    raise FileNotFoundError(
        "Could not infer Piper config JSON. Provide --piper-config explicitly."
    )


def load_piper_voice(
    model_path: Path,
    config_path: Path | None,
    *,
    length_scale: float | None = None,
    noise_scale: float | None = None,
    noise_w: float | None = None,
) -> PiperVoice:
    resolved = resolve_piper_config_path(model_path, config_path)
    with open(resolved, "r", encoding="utf-8") as f:
        config_data = json.load(f)

    overrides = {
        "length_scale": length_scale,
        "noise_scale": noise_scale,
        "noise_w": noise_w,
    }

    applied = {k: v for k, v in overrides.items() if v is not None}
    tmp_path: Path | None = None

    if applied:
        config_data.update(applied)
        tmp_file = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        tmp_path = Path(tmp_file.name)
        json.dump(config_data, tmp_file)
        tmp_file.flush()
        tmp_file.close()
        config_to_use = tmp_path
        print(
            "Loading Piper voice '{}' with overrides {}".format(
                model_path.name,
                ", ".join(f"{k}={v}" for k, v in applied.items()),
            ),
            file=sys.stderr,
        )
    else:
        config_to_use = resolved
        print(
            f"Loading Piper voice '{model_path.name}' with config '{resolved.name}'…",
            file=sys.stderr,
        )

    try:
        voice = PiperVoice.load(str(model_path), config_path=str(config_to_use))
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)

    return voice


def ensure_log_dir() -> Path:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    return LOG_ROOT


def append_log_line(log_path: Path, payload: dict[str, Any]) -> None:
    record = {"timestamp": datetime.now().isoformat(), **payload}
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def find_input_device(keyword: str, min_channels: int = 1) -> int | None:
    keyword_lower = keyword.lower()
    for idx, device in enumerate(sd.query_devices()):
        name = device.get("name", "")
        if keyword_lower in name.lower() and device.get("max_input_channels", 0) >= min_channels:
            return idx
    return None


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
        device=config.input_device_index,
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
    voice: PiperVoice, text: str, output_wav: Path
) -> None:
    print("Synthesizing speech with Piper…", file=sys.stderr)
    audio_iter = voice.synthesize(text)
    base_sample_rate = int(
        getattr(voice, "sample_rate", getattr(getattr(voice, "config", {}), "sample_rate", DEFAULT_SAMPLE_RATE))
    )

    def extract_audio_field(obj: Any) -> Any | None:
        field_candidates = (
            "audio",
            "_audio",
            "buffer",
            "data",
            "pcm",
            "samples",
            "wave",
            "waveform",
            "frames",
            "chunk",
            "audio_int16_bytes",
            "audio_int16_array",
            "audio_float_array",
            "_audio_int16_bytes",
            "_audio_int16_array",
        )
        for attr in field_candidates:
            value = getattr(obj, attr, None)
            if value is not None:
                return value
        return None

    def to_bytes_and_rate(chunk: Any) -> tuple[bytes, int | None]:
        current_rate: int | None = None
        data: Any = chunk

        if AudioChunk is not None and isinstance(chunk, AudioChunk):
            maybe = extract_audio_field(chunk)
            if maybe is not None:
                data = maybe
            current_rate = getattr(chunk, "sample_rate", None)
        elif isinstance(chunk, dict):
            if "audio" in chunk:
                data = chunk["audio"]
            else:
                for key in ("buffer", "data", "pcm", "samples"):
                    if key in chunk:
                        data = chunk[key]
                        break
            current_rate = chunk.get("sample_rate")
        else:
            maybe = extract_audio_field(chunk)
            if maybe is not None:
                data = maybe
                current_rate = getattr(chunk, "sample_rate", None)

        if isinstance(data, np.ndarray):
            return data.astype(np.int16).tobytes(), current_rate
        if isinstance(data, (bytes, bytearray, memoryview)):
            return bytes(data), current_rate
        if isinstance(data, (tuple, list)) and data:
            first = data[0]
            if isinstance(first, np.ndarray):
                return first.astype(np.int16).tobytes(), current_rate
            if isinstance(first, (bytes, bytearray, memoryview)):
                return bytes(first), current_rate
        if data is chunk and hasattr(chunk, "__iter__") and not isinstance(
            chunk, (str, bytes, bytearray, memoryview)
        ):
            try:
                arr = np.fromiter(chunk, dtype=np.int16)
                return arr.tobytes(), current_rate
            except TypeError:
                pass

        # Fall back to generic bytes conversion if possible
        try:
            return bytes(data), current_rate
        except Exception as exc:
            raise TypeError(
                f"Unsupported Piper chunk type: {type(chunk)!r} (available attrs: {dir(chunk)})"
            ) from exc

    with wave.open(str(output_wav), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # 16-bit PCM
        wav_file.setframerate(base_sample_rate)

        for chunk in audio_iter:
            data, maybe_rate = to_bytes_and_rate(chunk)
            if maybe_rate and maybe_rate != base_sample_rate:
                wav_file.setframerate(maybe_rate)
            wav_file.writeframes(data)


def play_audio(audio_path: Path) -> None:
    data, samplerate = sf.read(audio_path, dtype="float32")
    sd.play(data, samplerate)
    sd.wait()


def build_initial_messages(system_prompt: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": system_prompt}]


def run_conversation(config: ConversationConfig) -> None:
    whisper_model = load_whisper_model(config.whisper_model)
    messages = build_initial_messages(config.system_prompt)
    assert config.piper_voice is not None
    piper_voice = load_piper_voice(
        config.piper_voice,
        config.piper_config,
        length_scale=config.piper_length_scale,
        noise_scale=config.piper_noise_scale,
        noise_w=config.piper_noise_w,
    )

    log_dir = ensure_log_dir()
    session_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = log_dir / f"voice-session-{session_id}.jsonl"
    append_log_line(
        log_file,
        {
            "type": "session_start",
            "session_id": session_id,
            "config": {
                "ollama_model": config.ollama_model,
                "whisper_model": config.whisper_model,
                "piper_voice": str(config.piper_voice),
                "piper_config": str(config.piper_config) if config.piper_config else None,
                "length_scale": config.piper_length_scale,
                "noise_scale": config.piper_noise_scale,
                "noise_w": config.piper_noise_w,
                "sample_rate": config.sample_rate,
            },
        },
    )

    if config.input_device_keyword:
        device_index = find_input_device(config.input_device_keyword)
        if device_index is not None:
            config.input_device_index = device_index
            dev_info = sd.query_devices(device_index)
            print(
                f"Using input device #{device_index}: {dev_info['name']}",
                file=sys.stderr,
            )
        else:
            print(
                f"Warning: no input device found matching '{config.input_device_keyword}'."
                " Falling back to system default.",
                file=sys.stderr,
            )

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
                append_log_line(
                    log_file,
                    {
                        "type": "user",
                        "session_id": session_id,
                        "turn": turn,
                        "text": user_text,
                        "audio_path": str(raw_audio),
                    },
                )
                assistant_text = query_ollama(config.ollama_model, messages)
                if not assistant_text:
                    print("Assistant returned empty response; stopping.")
                    break

                messages.append({"role": "assistant", "content": assistant_text})

                response_audio = tmpdir_path / f"turn-{turn:03d}-response.wav"
                synthesize_with_piper(
                    piper_voice,
                    assistant_text,
                    response_audio,
                )
                play_audio(response_audio)
                append_log_line(
                    log_file,
                    {
                        "type": "assistant",
                        "session_id": session_id,
                        "turn": turn,
                        "text": assistant_text,
                        "audio_path": str(response_audio),
                    },
                )

                turn += 1
        except KeyboardInterrupt:
            print("\nExiting conversation.")
        finally:
            append_log_line(
                log_file,
                {"type": "session_end", "session_id": session_id},
            )


def main() -> None:
    config = parse_args()
    run_conversation(config)


if __name__ == "__main__":
    main()

