from pathlib import Path
import sys

import numpy as np

# ensure repository root on path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from app.core.mapping import (
    EnvelopeFollower,
    gamma_curve,
    mix_beat_energy,
    resample_series,
    map_image_scale,
    map_ring_radius,
)


def test_envelope_follower_attack_release():
    x = np.concatenate([np.zeros(5), np.ones(5), np.zeros(5)])

    env_fast = EnvelopeFollower(attack_ms=50, release_ms=100).process(x)
    env_slow = EnvelopeFollower(attack_ms=150, release_ms=100).process(x)
    assert env_fast[5] > env_slow[5]

    rel_slow = EnvelopeFollower(attack_ms=50, release_ms=300).process(x)
    rel_fast = EnvelopeFollower(attack_ms=50, release_ms=100).process(x)
    assert rel_slow[10] > rel_fast[10]


def test_gamma_curve():
    x = np.array([0.25, 0.5, 0.75])
    y = gamma_curve(x, 2.0)
    assert np.all(np.diff(y) > 0)
    assert np.all(y < x)


def test_mix_beat_energy():
    fps = 24.0
    length = int(fps * 2)
    beats = [0.0, 0.5, 1.0]
    energy = np.zeros(length)
    mix = mix_beat_energy(energy, beats, fps, k_beat=1.0)
    for bt in beats:
        idx = int(round(bt * fps))
        assert mix[idx] > 0.8


def test_resample_series():
    x = np.arange(10, dtype=float)
    y = resample_series(x, src_hop=512, sr=44100, dst_fps=24, length=24)
    assert len(y) == 24
    assert np.all(np.diff(y) >= 0)


def test_mapping_helpers():
    scale = map_image_scale(np.array([0.0, 1.0]), k=0.2)
    assert np.allclose(scale, np.array([1.0, 1.2]))

    radius = map_ring_radius(0.35, np.array([0.0, 1.0]), k=0.2)
    assert np.allclose(radius, np.array([0.35, 0.55]))

