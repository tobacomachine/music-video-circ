from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.core.analysis import analyze


def _parse_bands(spec: str) -> list[tuple[int, int]]:
    bands: list[tuple[int, int]] = []
    for part in spec.split(","):
        low_str, high_str = part.split("-")
        bands.append((int(low_str), int(high_str)))
    return bands


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze audio file")
    parser.add_argument("--audio", required=True, help="Path to audio file")
    parser.add_argument("--sr", type=int, default=44100)
    parser.add_argument("--n_fft", type=int, default=2048)
    parser.add_argument("--hop", type=int, default=512)
    parser.add_argument(
        "--bands",
        type=str,
        default="20-160,160-2000,2000-16000",
        help="Comma-separated frequency bands (low-high)",
    )
    parser.add_argument("--out", type=str, help="Output JSON path")
    args = parser.parse_args()

    params = {
        "sr": args.sr,
        "n_fft": args.n_fft,
        "hop": args.hop,
        "bands": _parse_bands(args.bands),
        "beat_track": True,
    }
    result = analyze(args.audio, params)
    output = json.dumps(result, indent=2)
    if args.out:
        Path(args.out).write_text(output, encoding="utf-8")
    else:
        print(output)


if __name__ == "__main__":
    main()
