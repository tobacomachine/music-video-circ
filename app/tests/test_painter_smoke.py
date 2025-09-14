"""Smoke tests for :class:`Painter`."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))

from app.core.visual_engine.painter import Painter


def test_basic_render_smoke() -> None:
    painter = Painter(640, 360, headless=True)
    painter.begin_frame()
    painter.clear("#000")
    painter.fill_linear_gradient(["#112233", "#334455"], 45)
    painter.stroke_circle_ndc(radius_ndc=0.4, thickness_ndc=0.05, color="#AABBCC", alpha=0.8)
    painter.draw_radial_bars(
        base_radius_ndc=0.5,
        values=np.linspace(0, 1, 32),
        thickness_ndc=0.15,
        roundness=0.5,
    )
    frame = painter.end_frame()
    assert isinstance(frame, np.ndarray)
    assert frame.shape == (360, 640, 3)
    assert frame.mean() > 0

