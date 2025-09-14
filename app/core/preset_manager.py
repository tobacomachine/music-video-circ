"""Preset management and schema definitions for presets."""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Tuple, List, Dict, Any
import json
import datetime as _dt
import re

from pydantic import BaseModel, Field, field_validator, model_validator

__all__ = [
    "normalize_hex",
    "rgb_to_hex",
    "Resolution",
    "OutputCfg",
    "BackgroundAnimKF",
    "BackgroundReactivity",
    "BackgroundAnim",
    "BackgroundCfg",
    "VisualMappingCfg",
    "RingCfg",
    "BarsCfg",
    "VisualColorCfg",
    "VisualModeCfg",
    "VisualCfg",
    "CenterImageReactivity",
    "CenterImageCfg",
    "AudioAnalysisCfg",
    "AudioCfg",
    "Meta",
    "PresetModel",
    "example_default",
    "PresetManager",
]

_HEX_RE = re.compile(r"^[0-9a-fA-F]{6}$")
_RGB_RE = re.compile(r"rgb\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})\s*\)", re.I)


def rgb_to_hex(s: str) -> str:
    """Convert an ``rgb(r,g,b)`` string to ``#rrggbb``."""
    m = _RGB_RE.fullmatch(s.strip())
    if not m:
        raise ValueError(f"Invalid rgb color: {s!r}")
    r, g, b = (max(0, min(int(v), 255)) for v in m.groups())
    return f"#{r:02x}{g:02x}{b:02x}"


def normalize_hex(s: str) -> str:
    """Normalize a color string to ``#rrggbb``.

    Accepts ``rgb(r,g,b)`` or hex strings with or without ``#``.
    """
    s = s.strip().lower()
    if s.startswith("rgb"):
        return rgb_to_hex(s)
    if s.startswith("#"):
        s = s[1:]
    if len(s) == 3:
        s = "".join(ch * 2 for ch in s)
    if not _HEX_RE.fullmatch(s):
        raise ValueError(f"Invalid hex color: #{s}")
    return f"#{s}"


class Resolution(BaseModel):
    width: int = 1920
    height: int = 1080

    @field_validator("width", "height")
    @classmethod
    def _min_size(cls, v: int) -> int:
        return max(16, int(v))


class OutputCfg(BaseModel):
    resolution: Resolution = Resolution()
    fps: int = 24
    aspect_mode: Literal["fit", "fill", "stretch"] = "fit"
    profile: str = "youtube_1080p24"

    @field_validator("fps")
    @classmethod
    def _clamp_fps(cls, v: int) -> int:
        return max(1, min(int(v), 120))


class BackgroundAnimKF(BaseModel):
    t: float
    colors: Optional[List[str]] = None
    angle: Optional[float] = None
    center: Optional[Tuple[float, float]] = None

    @field_validator("colors", mode="before")
    @classmethod
    def _norm_colors(cls, v):
        if v is None:
            return v
        return [normalize_hex(c) for c in v]


class BackgroundReactivity(BaseModel):
    target: Literal["bass", "mid", "treble", "global"] = "bass"
    intensity: float = 0.6


class BackgroundAnim(BaseModel):
    speed: float = 0.2
    keyframes: List[BackgroundAnimKF] = Field(default_factory=list)


class BackgroundCfg(BaseModel):
    type: Literal["solid", "gradient", "gradient_anim", "gradient_dynamic"] = "gradient_dynamic"
    colors: List[str] = Field(default_factory=lambda: ["#0f0f1a", "#101a2b", "#192a56"])
    angle: float = 45.0
    anim: Optional[BackgroundAnim] = None
    reactivity: Optional[BackgroundReactivity] = BackgroundReactivity()

    @field_validator("colors", mode="before")
    @classmethod
    def _norm_colors(cls, v):
        return [normalize_hex(c) for c in v]

    @model_validator(mode="after")
    def _check_colors(self) -> "BackgroundCfg":
        if self.type == "solid" and len(self.colors) < 1:
            raise ValueError("solid background requires at least 1 color")
        if self.type != "solid" and len(self.colors) < 2:
            raise ValueError("gradient backgrounds require at least 2 colors")
        return self


class VisualMappingCfg(BaseModel):
    attack_ms: int = 80
    release_ms: int = 220
    sensitivity: float = 0.8
    threshold: float = 0.15


class RingCfg(BaseModel):
    base_radius: float = 0.35
    thickness: float = 0.02
    glow: float = 0.4

    @field_validator("glow")
    @classmethod
    def _clamp_glow(cls, v: float) -> float:
        return max(0.0, min(float(v), 1.0))


class BarsCfg(BaseModel):
    count: int = 96
    scale: float = 0.25
    distribution: Literal["log", "linear"] = "log"
    roundness: float = 0.4

    @field_validator("count")
    @classmethod
    def _clamp_count(cls, v: int) -> int:
        return max(1, min(int(v), 512))

    @field_validator("roundness")
    @classmethod
    def _clamp_roundness(cls, v: float) -> float:
        return max(0.0, min(float(v), 1.0))


class VisualColorCfg(BaseModel):
    palette: str = "auto"
    gamma: float = 1.2

    @field_validator("gamma")
    @classmethod
    def _clamp_gamma(cls, v: float) -> float:
        return max(0.0, min(float(v), 3.0))


class VisualModeCfg(BaseModel):
    ring: bool = True
    bars: bool = True


class VisualCfg(BaseModel):
    mode: VisualModeCfg = Field(default_factory=VisualModeCfg)
    ring: RingCfg = Field(default_factory=RingCfg)
    bars: BarsCfg = Field(default_factory=BarsCfg)
    color: VisualColorCfg = Field(default_factory=VisualColorCfg)
    mapping: VisualMappingCfg = Field(default_factory=VisualMappingCfg)


class CenterImageReactivity(BaseModel):
    scale_on_beat: float = 0.15
    rotate_per_sec: float = 5.0
    shake: float = 0.0
    bloom: float = 0.2


class CenterImageCfg(BaseModel):
    path: Optional[str] = None
    reactivity: CenterImageReactivity = CenterImageReactivity()


class AudioAnalysisCfg(BaseModel):
    sr: int = 44100
    n_fft: int = 2048
    hop: int = 512
    bands: List[Tuple[int, int]] = Field(default_factory=lambda: [(20, 160), (160, 2000), (2000, 16000)])
    beat_track: bool = True


class AudioCfg(BaseModel):
    normalize: bool = True
    analysis: AudioAnalysisCfg = AudioAnalysisCfg()


class Meta(BaseModel):
    name: str
    author: Optional[str] = ""
    created: Optional[str] = None

    @model_validator(mode="after")
    def _ensure_created(self) -> "Meta":
        if not self.created:
            self.created = _dt.date.today().isoformat()
        else:
            try:
                _dt.date.fromisoformat(self.created)
            except ValueError as exc:  # pragma: no cover - defensive
                raise ValueError("meta.created must be ISO 8601 date") from exc
        return self


class PresetModel(BaseModel):
    schema_version: int = 1
    meta: Meta
    output: OutputCfg = OutputCfg()
    background: BackgroundCfg = BackgroundCfg()
    visual: VisualCfg = VisualCfg()
    center_image: CenterImageCfg = CenterImageCfg()
    audio: AudioCfg = AudioCfg()


def example_default() -> PresetModel:
    """Return an example preset with default values."""
    return PresetModel(meta=Meta(name="example"))


def _deep_update(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in update.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


class PresetManager:
    """Load and save preset models."""

    @staticmethod
    def load(path: str | Path) -> PresetModel:
        p = Path(path)
        try:
            raw = json.loads(p.read_text("utf-8"))
        except FileNotFoundError as exc:  # pragma: no cover - defensive
            raise FileNotFoundError(f"Preset file not found: {p}") from exc
        if raw.get("schema_version", 0) < 1:
            raw = PresetManager.migrate(raw)
        model = PresetModel.model_validate(raw)
        return model

    @staticmethod
    def save(model: PresetModel, path: str | Path) -> None:
        p = Path(path)
        data = model.model_dump()
        p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def migrate(raw: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(raw)
        if data.get("schema_version", 0) < 1:
            data["schema_version"] = 1
        if "bg" in data and "background" not in data:
            data["background"] = data.pop("bg")
        if "vis" in data and "visual" not in data:
            data["visual"] = data.pop("vis")
        defaults = example_default().model_dump()
        return _deep_update(defaults, data)
