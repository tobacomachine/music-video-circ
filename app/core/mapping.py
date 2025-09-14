"""Signal mapping utilities.

Pure numerical helpers used to map analysis data to values suitable for
visual elements.  All operations are deterministic and implemented with
NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

__all__ = [
    "EnvelopeFollower",
    "apply_envelope",
    "gamma_curve",
    "mix_beat_energy",
    "map_ring_radius",
    "map_ring_thickness",
    "map_bar_height",
    "map_image_scale",
    "resample_series",
]


@dataclass
class EnvelopeFollower:
    """Simple attack/release envelope follower.

    The state ``y`` is updated per hop using::

        y[n] = (1 - a) * x[n] + a * y[n-1]   if x[n] > y[n-1]
        y[n] = (1 - r) * x[n] + r * y[n-1]   otherwise

    where ``a`` and ``r`` are IIR coefficients derived from the attack and
    release times in milliseconds.
    """

    attack_ms: int = 80
    release_ms: int = 220
    sr: int = 44100
    hop: int = 512

    def __post_init__(self) -> None:
        attack_ms = max(float(self.attack_ms), 1e-3)
        release_ms = max(float(self.release_ms), 1e-3)
        hop = max(int(self.hop), 1)
        sr = max(int(self.sr), 1)

        self._attack = float(np.exp(-hop / (sr * (attack_ms / 1000.0))))
        self._release = float(np.exp(-hop / (sr * (release_ms / 1000.0))))
        self.reset()

    def reset(self) -> None:
        """Reset internal state to zero."""

        self._value = 0.0

    def process(self, x: np.ndarray) -> np.ndarray:
        """Process a sequence ``x`` and return the envelope.

        Parameters
        ----------
        x:
            Input array. Values are clamped to ``[0, 1]`` to avoid NaNs or
            runaway values.
        """

        data = np.asarray(x, dtype=float)
        data = np.clip(np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
        env = np.empty_like(data)

        y = float(self._value)
        for i, xi in enumerate(data):
            coeff = self._attack if xi > y else self._release
            y = coeff * y + (1.0 - coeff) * xi
            env[i] = y

        self._value = y
        return env


def apply_envelope(
    x: np.ndarray,
    attack_ms: int = 80,
    release_ms: int = 220,
    sr: int = 44100,
    hop: int = 512,
) -> np.ndarray:
    """Convenience wrapper returning ``EnvelopeFollower(...).process(x)``."""

    return EnvelopeFollower(attack_ms, release_ms, sr, hop).process(x)


def gamma_curve(x: np.ndarray, gamma: float = 1.2) -> np.ndarray:
    """Apply ``x^gamma`` after clamping ``x`` to ``[0, 1]``."""

    gamma = max(float(gamma), 1e-6)
    return np.power(np.clip(x, 0.0, 1.0), gamma)


def mix_beat_energy(
    energy_global: np.ndarray,
    beats_times: List[float],
    fps: float,
    k_beat: float = 0.5,
    width_ms: float = 90.0,
    length: int | None = None,
) -> np.ndarray:
    """Mix global energy with Gaussian beat pulses.

    Pulses are centred at each beat time with ``sigma = width_ms / 1000 * fps / 2.355``.
    The pulses are normalised to ``[0, 1]`` and linearly blended with
    ``energy_global`` using weight ``k_beat``.
    """

    energy_global = np.asarray(energy_global, dtype=float)
    if length is None:
        length = energy_global.shape[0]

    if energy_global.shape[0] != length:
        # Resample energy to the desired length
        src = np.linspace(0.0, 1.0, energy_global.shape[0])
        dst = np.linspace(0.0, 1.0, length)
        energy = np.interp(dst, src, energy_global)
    else:
        energy = energy_global

    pulses = np.zeros(length, dtype=float)
    sigma = width_ms / 1000.0 * fps / 2.355
    sigma = max(sigma, 1e-6)
    t = np.arange(length)
    for bt in beats_times:
        centre = bt * fps
        pulses += np.exp(-0.5 * ((t - centre) / sigma) ** 2)

    if pulses.max() > 0:
        pulses /= pulses.max()

    mix = np.clip((1.0 - k_beat) * energy + k_beat * pulses, 0.0, 1.0)
    return mix


def map_ring_radius(base: float, bass_energy: np.ndarray, k: float = 0.15) -> np.ndarray:
    """Return ``base + k * bass_energy`` clamped to ``[0, 1]``."""

    bass_energy = np.clip(bass_energy, 0.0, 1.0)
    return np.clip(base + k * bass_energy, 0.0, 1.0)


def map_ring_thickness(
    base: float, global_energy: np.ndarray, k: float = 0.1
) -> np.ndarray:
    """Return ``base + k * global_energy`` clamped to ``[0, 1]``."""

    global_energy = np.clip(global_energy, 0.0, 1.0)
    return np.clip(base + k * global_energy, 0.0, 1.0)


def map_bar_height(
    energy_bins: np.ndarray, k: float = 0.25, gamma: float = 1.2
) -> np.ndarray:
    """Gamma-correct ``energy_bins`` and scale by ``k`` (clamped to ``[0, 1]``)."""

    return np.clip(k * gamma_curve(energy_bins, gamma), 0.0, 1.0)


def map_image_scale(beat_mix: np.ndarray, k: float = 0.15) -> np.ndarray:
    """Return ``1 + k * beat_mix`` (beat mix is clamped to ``[0, 1]``)."""

    beat_mix = np.clip(beat_mix, 0.0, 1.0)
    return 1.0 + k * beat_mix


def resample_series(
    x: np.ndarray,
    src_hop: int,
    sr: int,
    dst_fps: float,
    length: int,
) -> np.ndarray:
    """Linearly interpolate ``x`` from hop ticks to ``dst_fps`` ticks."""

    x = np.asarray(x, dtype=float)
    t_src = np.arange(x.shape[0]) * (src_hop / float(sr))
    t_dst = np.arange(length) / float(dst_fps)
    return np.interp(t_dst, t_src, x, left=x[0], right=x[-1])

