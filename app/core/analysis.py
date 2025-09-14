from __future__ import annotations

from pathlib import Path
from typing import Any

import librosa
import numpy as np

# --- Audio loading -------------------------------------------------------

def load_audio(path: str | Path, sr: int = 44100) -> np.ndarray:
    """Load an audio file as mono ``float32``.

    The signal is normalized if its peak amplitude exceeds 1.0.
    """
    file = Path(path)
    if not file.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    try:
        y, _ = librosa.load(file, sr=sr, mono=True)
    except Exception as exc:  # pragma: no cover - delegated to librosa
        raise ValueError(f"Could not load audio: {exc}") from exc
    y = y.astype(np.float32, copy=False)
    peak = float(np.max(np.abs(y)))
    if peak > 1.0:
        y /= peak
    return y


# --- Spectral analysis ---------------------------------------------------

def compute_stft(
    y: np.ndarray, sr: int, n_fft: int = 2048, hop: int = 512
) -> np.ndarray:
    """Compute magnitude STFT."""
    return np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))


def band_energy(
    S: np.ndarray, sr: int, bands: list[tuple[int, int]]
) -> dict[str, np.ndarray]:
    """Return average energy per frequency band and global mean."""
    n_fft = (S.shape[0] - 1) * 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    energy: dict[str, np.ndarray] = {}
    for label, (low, high) in zip(["bass", "mid", "treble"], bands):
        idx = np.where((freqs >= low) & (freqs < high))[0]
        if idx.size == 0:
            energy[label] = np.zeros(S.shape[1], dtype=float)
        else:
            energy[label] = np.mean(S[idx, :], axis=0)
    energy["global"] = np.mean(S, axis=0)
    return energy


# --- Beat / onset detection ----------------------------------------------

def detect_beats(y: np.ndarray, sr: int, hop: int) -> dict[str, Any]:
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env, sr=sr, hop_length=hop
    )
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop)
    return {
        "bpm": float(tempo) if tempo else None,
        "beat_frames": beat_frames.tolist(),
        "beat_times": beat_times.tolist(),
        "onset_envelope": onset_env.tolist(),
    }


# --- Full pipeline -------------------------------------------------------

def analyze(path: str | Path, params: dict | None = None) -> dict[str, Any]:
    """Analyze an audio file and return metrics suitable for JSON output."""
    cfg = {
        "sr": 44100,
        "n_fft": 2048,
        "hop": 512,
        "bands": [(20, 160), (160, 2000), (2000, 16000)],
        "beat_track": True,
    }
    if params:
        cfg.update(params)
    y = load_audio(path, sr=cfg["sr"])
    S = compute_stft(y, cfg["sr"], cfg["n_fft"], cfg["hop"])
    energies = band_energy(S, cfg["sr"], cfg["bands"])

    result: dict[str, Any] = {
        "sr": cfg["sr"],
        "n_fft": cfg["n_fft"],
        "hop": cfg["hop"],
        "frames": int(S.shape[1]),
        "duration": float(len(y) / cfg["sr"]),
        "bpm": None,
        "beat_frames": [],
        "beat_times": [],
        "onset_env": [],
        "energy": {k: v.tolist() for k, v in energies.items()},
    }
    if cfg.get("beat_track"):
        beats = detect_beats(y, cfg["sr"], cfg["hop"])
        result["bpm"] = beats["bpm"]
        result["beat_frames"] = beats["beat_frames"]
        result["beat_times"] = beats["beat_times"]
        result["onset_env"] = beats["onset_envelope"]
    return result
