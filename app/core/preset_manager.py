from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

HEX_COLOR_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")


def ensure_hex_color(value: str) -> str:
    """Validate and normalize a hex color."""
    if not HEX_COLOR_RE.match(value):
        raise ValueError(f"Invalid hex color: {value}")
    return value.upper()


class Meta(BaseModel):
    name: str
    author: str | None = None
    created: date | str


class Resolution(BaseModel):
    width: int
    height: int


class Output(BaseModel):
    resolution: Resolution
    fps: int = 24
    aspect_mode: Literal["fit", "fill", "stretch"] = "fit"
    profile: str


class BackgroundAnim(BaseModel):
    speed: float
    keyframes: list


class BackgroundReactivity(BaseModel):
    target: Literal["bass", "mid", "treble", "global"]
    intensity: float


class Background(BaseModel):
    type: Literal["solid", "gradient", "gradient_anim", "gradient_dynamic"]
    colors: list[str]
    angle: float | int = 0.0
    anim: BackgroundAnim | None = None
    reactivity: BackgroundReactivity | None = None

    @field_validator("colors")
    @classmethod
    def validate_colors(cls, v: list[str]) -> list[str]:
        if not 2 <= len(v) <= 4:
            raise ValueError("colors must contain between 2 and 4 items")
        return [ensure_hex_color(color) for color in v]


class VisualMode(BaseModel):
    ring: bool
    bars: bool


class VisualRing(BaseModel):
    base_radius: float
    thickness: float
    glow: float


class VisualBars(BaseModel):
    count: int
    scale: float
    distribution: Literal["log", "linear"]
    roundness: float


class VisualColor(BaseModel):
    palette: str | Literal["auto"]
    gamma: float


class VisualMapping(BaseModel):
    attack_ms: int
    release_ms: int
    sensitivity: float
    threshold: float


class Visual(BaseModel):
    mode: VisualMode
    ring: VisualRing
    bars: VisualBars
    color: VisualColor
    mapping: VisualMapping


class CenterReactivity(BaseModel):
    scale_on_beat: float
    rotate_per_sec: float
    shake: float
    bloom: float


class CenterImage(BaseModel):
    path: str | None = None
    reactivity: CenterReactivity | None = None


class AudioAnalysis(BaseModel):
    sr: int = 44100
    n_fft: int = 2048
    hop: int = 512
    bands: list[tuple[int, int]]
    beat_track: bool


class Audio(BaseModel):
    normalize: bool
    analysis: AudioAnalysis


class Preset(BaseModel):
    schema_version: int = Field(alias="schema_version")
    meta: Meta
    output: Output
    background: Background
    visual: Visual
    center_image: CenterImage
    audio: Audio

    @field_validator("schema_version")
    @classmethod
    def check_version(cls, v: int) -> int:
        if v != 1:
            raise ValueError("schema_version must be 1")
        return v

    @classmethod
    def from_dict(cls, data: dict) -> "Preset":
        return cls.model_validate(data)

    def to_dict(self) -> dict:
        return self.model_dump(mode="python", by_alias=True)


class PresetManager:
    @staticmethod
    def validate(preset_dict: dict) -> Preset:
        try:
            return Preset.from_dict(preset_dict)
        except ValidationError as exc:
            raise exc

    @staticmethod
    def migrate(preset_dict: dict) -> dict:
        version = preset_dict.get("schema_version")
        if version == 1:
            return dict(preset_dict)
        raise ValueError(f"Unsupported schema version: {version}")

    @classmethod
    def load(cls, path: str | Path) -> Preset:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        migrated = cls.migrate(data)
        return cls.validate(migrated)

    @staticmethod
    def save(preset: Preset, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(preset.to_dict(), fh, indent=2, sort_keys=True)
