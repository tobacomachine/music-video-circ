from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def resolve_ffmpeg() -> str:
    """Resolve path to ffmpeg executable.

    Looks for a bundled binary under ``./ffmpeg`` at the project root.
    If not found, attempts to use ``ffmpeg`` from ``PATH`` and validates
    it via ``ffmpeg -version``.
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    binary = "ffmpeg.exe" if sys.platform.startswith("win") else "ffmpeg"
    local_ffmpeg = project_root / "ffmpeg" / binary

    cmd = str(local_ffmpeg) if local_ffmpeg.exists() else shutil.which("ffmpeg")

    if not cmd:
        raise FileNotFoundError("ffmpeg executable not found")

    try:
        subprocess.run([cmd, "-version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception as exc:  # pragma: no cover - basic validation
        raise FileNotFoundError("ffmpeg executable not found or is invalid") from exc

    return cmd
