
from __future__ import annotations

import argparse
import json

from app.ui.main_window import main as ui_main


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", help="Preset path", default="")
    parser.add_argument("--analyze", help="Audio file to analyze")
    args = parser.parse_args()

    if args.analyze:
        from app.core.analysis import analyze

        result = analyze(args.analyze, None)
        print(json.dumps(result, indent=2))
        return
    if args.preset:
        print(f"Preset: {args.preset}")
    ui_main()

=======
from app.ui.main_window import main

if __name__ == "__main__":
    main()
