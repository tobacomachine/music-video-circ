"""Basic rendering context and helpers for the preview engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from app.core.analysis import AnalysisResult
    from app.core.preset_manager import PresetModel


@dataclass(slots=True)
class RenderCtx:
    """Context information passed to render layers."""

    width: int
    height: int
    t_sec: float
    fps: float
    analysis: "AnalysisResult"
    preset: "PresetModel"


def aspect_min(ctx: RenderCtx) -> float:
    """Return the smallest dimension of ``ctx`` in pixels."""

    return float(min(ctx.width, ctx.height))


def ndc_to_px(ctx: RenderCtx, x: float, y: float) -> tuple[int, int]:
    """Convert normalised device coordinates to pixel coordinates."""

    half = aspect_min(ctx) / 2.0
    px = ctx.width / 2.0 + x * half
    py = ctx.height / 2.0 - y * half
    return int(round(px)), int(round(py))


__all__ = ["RenderCtx", "aspect_min", "ndc_to_px"]

