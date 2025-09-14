from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from app.core.analysis import analyze


def _sine(freq: float, sr: int = 44100, duration: float = 2.0) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return np.sin(2 * np.pi * freq * t).astype("float32")


def test_band_energies(tmp_path: Path) -> None:
    sr = 44100
    low = _sine(60, sr)
    path_low = tmp_path / "low.wav"
    sf.write(path_low, low, sr)
    res_low = analyze(str(path_low), None)
    frames = res_low["frames"]
    assert frames > 0
    assert all(len(res_low["energy"][k]) == frames for k in ["bass", "mid", "treble", "global"])
    assert len(res_low["onset_env"]) == frames
    bass_mean = float(np.mean(res_low["energy"]["bass"]))
    treble_mean = float(np.mean(res_low["energy"]["treble"]))
    assert bass_mean > treble_mean * 10

    high = _sine(4000, sr)
    path_high = tmp_path / "high.wav"
    sf.write(path_high, high, sr)
    res_high = analyze(str(path_high), None)
    bass_mean_high = float(np.mean(res_high["energy"]["bass"]))
    treble_mean_high = float(np.mean(res_high["energy"]["treble"]))
    assert treble_mean_high > bass_mean_high * 10


def test_detect_beats(tmp_path: Path) -> None:
    sr = 44100
    duration = 2.0
    y = np.zeros(int(sr * duration), dtype="float32")
    step = int(0.5 * sr)
    for i in range(0, len(y), step):
        y[i:i + 100] = 1.0
    path = tmp_path / "click.wav"
    sf.write(path, y, sr)
    res = analyze(str(path), None)
    assert res["bpm"] is not None
    assert abs(res["bpm"] - 120) < 5


def test_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        analyze("does-not-exist.wav")
