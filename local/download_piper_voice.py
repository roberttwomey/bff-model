#!/usr/bin/env python3
"""Helper script around `python3 -m piper.download_voices`."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import urlopen

from piper import download_voices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "voice",
        nargs="?",
        help="Voice identifier such as 'en_US-lessac-medium'. Run with --list to see options.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to place the downloaded assets in (defaults to ./piper next to this script).",
    )
    parser.add_argument(
        "--force-redownload",
        action="store_true",
        help="Force re-download even if the files already exist.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available voices and exit.",
    )
    parser.add_argument(
        "--list-language",
        "--list-lang",
        metavar="LANG",
        help="List voices for a language code like 'en_GB' or 'en_US' and exit.",
    )
    return parser.parse_args()


def list_voices_for_language(language_code: str) -> list[str]:
    """Return voices that match a given language code (e.g. en_GB)."""
    language_code = language_code.strip()
    prefix = f"{language_code}-"

    with urlopen(download_voices.VOICES_JSON) as response:
        voices = json.load(response)

    return sorted(name for name in voices.keys() if name.startswith(prefix))


def main() -> None:
    args = parse_args()

    if args.list:
        download_voices.list_voices()
        return

    if args.list_language:
        matches = list_voices_for_language(args.list_language)
        if matches:
            for voice in matches:
                print(voice)
        else:
            print(f"No voices found for language '{args.list_language}'.")
        return

    if not args.voice:
        raise SystemExit("Please provide a voice name, e.g. `en_US-lessac-medium`.")

    if args.output_dir:
        download_dir = Path(args.output_dir)
    else:
        download_dir = Path(__file__).resolve().parent / "piper"

    download_dir.mkdir(parents=True, exist_ok=True)

    download_voices.download_voice(
        args.voice, download_dir, force_redownload=args.force_redownload
    )


if __name__ == "__main__":
    main()

