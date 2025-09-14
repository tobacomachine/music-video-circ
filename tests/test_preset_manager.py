from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.preset_manager import PresetManager

ASSETS = Path("assets/presets")


def test_load_minimal_ring(tmp_path: Path) -> None:
    preset = PresetManager.load(ASSETS / "minimal_ring.json")
    out = tmp_path / "round.json"
    PresetManager.save(preset, out)
    loaded = PresetManager.load(out)
    assert preset.to_dict() == loaded.to_dict()


def _load_dict() -> dict:
    with open(ASSETS / "minimal_ring.json", "r", encoding="utf-8") as fh:
        return json.load(fh)


def test_invalid_color() -> None:
    data = _load_dict()
    data["background"]["colors"][0] = "#GGGGGG"
    with pytest.raises(ValidationError):
        PresetManager.validate(data)


def test_color_count() -> None:
    data = _load_dict()
    data["background"]["colors"] = ["#000000"]
    with pytest.raises(ValidationError):
        PresetManager.validate(data)


def test_invalid_distribution() -> None:
    data = _load_dict()
    data["visual"]["bars"]["distribution"] = "invalid"
    with pytest.raises(ValidationError):
        PresetManager.validate(data)
