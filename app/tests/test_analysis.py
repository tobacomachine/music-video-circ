from pathlib import Path
import sys

import numpy as np
import soundfile as sf

# ensure repository root on path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from app.core.analysis import AnalysisConfig, analyze_audio


SR = 44100


def test_sine_low_band(tmp_path):
    duration = 2.0
    t = np.arange(int(SR * duration)) / SR
    y = 0.5 * np.sin(2 * np.pi * 60 * t)
    path = tmp_path / "sine.wav"
    sf.write(path, y, SR)
    result = analyze_audio(str(path), AnalysisConfig())
    means = result.energy_bands.mean(axis=1)
    assert means[0] > means[1]
    assert means[0] > means[2]


def test_click_track_bpm(tmp_path):
    duration = 4.0
    y = np.zeros(int(SR * duration), dtype=np.float32)
    for i in range(int(duration / 0.5)):
        y[int(i * 0.5 * SR)] = 1.0
    path = tmp_path / "click.wav"
    sf.write(path, y, SR)
    result = analyze_audio(str(path), AnalysisConfig())
    assert abs(result.bpm - 120) < 2
    assert len(result.beats_times) >= 6


def test_white_noise(tmp_path):
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, SR * 2).astype(np.float32)
    path = tmp_path / "noise.wav"
    sf.write(path, y, SR)
    result = analyze_audio(str(path), AnalysisConfig())
    assert result.energy_bands.shape[0] == 3
    assert np.all(result.energy_bands.mean(axis=1) > 0)
