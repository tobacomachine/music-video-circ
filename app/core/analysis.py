"""Audio analysis utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Union
import io
import os
import subprocess

import numpy as np
import soundfile as sf
import librosa


@dataclass
class AnalysisConfig:
    """Configuration for :func:`analyze_audio`."""

    sr: int = 44100
    n_fft: int = 2048
    hop: int = 512
    bands: Tuple[Tuple[float, float], ...] = (
        (20, 160),
        (160, 2000),
        (2000, 16000),
    )
    beat_track: bool = True


@dataclass
class AnalysisResult:
    """Result object returned by :func:`analyze_audio`."""

    sr: int
    hop: int
    fps_base: float
    duration: float
    bpm: float
    beats_times: List[float]
    energy_bands: np.ndarray
    energy_global: np.ndarray
    onsets_times: List[float]


def _load_audio(path_or_bytes: Union[str, os.PathLike, bytes]) -> Tuple[np.ndarray, int]:
    """Load audio using soundfile with ffmpeg fallback."""

    try:
        if isinstance(path_or_bytes, (str, os.PathLike)):
            data, sr = sf.read(path_or_bytes, always_2d=False, dtype="float32")
        else:
            data, sr = sf.read(io.BytesIO(path_or_bytes), always_2d=False, dtype="float32")
    except Exception:
        if isinstance(path_or_bytes, (str, os.PathLike)):
            cmd = ["ffmpeg", "-i", str(path_or_bytes), "-f", "wav", "-"]
            inp = None
        else:
            cmd = ["ffmpeg", "-i", "pipe:0", "-f", "wav", "-"]
            inp = path_or_bytes
        proc = subprocess.run(cmd, input=inp, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        data, sr = sf.read(io.BytesIO(proc.stdout), always_2d=False, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data, sr


def analyze_audio(path_or_bytes: Union[str, os.PathLike, bytes], config: AnalysisConfig) -> AnalysisResult:
    """Analyze an audio file and return an :class:`AnalysisResult`."""

    np.random.seed(0)

    y, sr = _load_audio(path_or_bytes)
    if sr != config.sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=config.sr)
        sr = config.sr

    duration = len(y) / sr
    stft = librosa.stft(y, n_fft=config.n_fft, hop_length=config.hop, window="hann")
    freqs = librosa.fft_frequencies(sr=sr, n_fft=config.n_fft)
    nbands = len(config.bands)
    frames = stft.shape[1]
    energy_bands = np.zeros((nbands, frames), dtype=np.float32)
    for i, (low, high) in enumerate(config.bands):
        mask = (freqs >= low) & (freqs < high)
        if np.any(mask):
            mag = np.abs(stft[mask, :])
            band = np.sqrt(np.mean(mag ** 2, axis=0))
        else:
            band = np.zeros(frames, dtype=np.float32)
        max_val = float(np.max(band))
        if max_val > 0:
            band /= max_val
        energy_bands[i] = band

    weights = np.array([hi - lo for lo, hi in config.bands], dtype=np.float32)
    energy_global = np.average(energy_bands, axis=0, weights=weights)

    onsets_times = librosa.onset.onset_detect(y=y, sr=sr, hop_length=config.hop, units="time").tolist()

    bpm = 0.0
    beats_times: List[float] = []
    if config.beat_track:
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=config.hop)
        bpm = float(np.asarray(tempo).squeeze())
        beats_times = librosa.frames_to_time(beats, sr=sr, hop_length=config.hop).tolist()

    fps_base = sr / config.hop
    return AnalysisResult(sr=sr, hop=config.hop, fps_base=fps_base, duration=duration,
                          bpm=bpm, beats_times=beats_times,
                          energy_bands=energy_bands, energy_global=energy_global,
                          onsets_times=onsets_times)


__all__ = ["AnalysisConfig", "AnalysisResult", "analyze_audio"]
