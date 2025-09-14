from pathlib import Path
import json
import datetime as dt

from app.core.preset_manager import PresetManager

ASSETS = Path(__file__).resolve().parents[2] / "app" / "assets"
PRESETS_DIR = ASSETS / "presets"
MIN_PRESET = PRESETS_DIR / "minimal_ring.json"


def test_minimal_preset_loads_ok():
    assert MIN_PRESET.exists(), f"No existe {MIN_PRESET}"
    model = PresetManager.load(str(MIN_PRESET))
    assert model.schema_version == 1
    assert model.visual.mode.ring or model.visual.mode.bars
    assert model.output.fps in (24, 30, 60)


def test_round_trip_save_load(tmp_path: Path):
    model = PresetManager.load(str(MIN_PRESET))
    out = tmp_path / "roundtrip.json"
    PresetManager.save(model, str(out))
    model2 = PresetManager.load(str(out))
    d1 = model.model_dump()
    d2 = model2.model_dump()
    # Permite variación en 'meta.created' si se autogenera
    if "meta" in d1 and "created" in d1["meta"]:
        d2["meta"]["created"] = d1["meta"]["created"]
    assert d1 == d2


def test_migration_from_legacy(tmp_path: Path):
    # Simula preset antiguo sin schema_version y con alias bg/vis
    legacy = {
        "meta": {
            "name": "Legacy",
            "author": "",
            "created": dt.date.today().isoformat(),
        },
        "output": {
            "resolution": {"width": 1920, "height": 1080},
            "fps": 24,
            "aspect_mode": "fit",
        },
        "bg": {"type": "gradient", "colors": ["#112233", "#445566"], "angle": 30},
        "vis": {
            "mode": {"ring": True, "bars": False},
            "ring": {"base_radius": 0.35, "thickness": 0.02, "glow": 0.4},
            "bars": {
                "count": 64,
                "scale": 0.25,
                "distribution": "log",
                "roundness": 0.4,
            },
            "color": {"palette": "auto", "gamma": 1.2},
            "mapping": {
                "attack_ms": 80,
                "release_ms": 220,
                "sensitivity": 0.8,
                "threshold": 0.15,
            },
        },
        "center_image": {
            "path": None,
            "reactivity": {
                "scale_on_beat": 0.15,
                "rotate_per_sec": 5,
                "shake": 0.0,
                "bloom": 0.2,
            },
        },
        "audio": {
            "normalize": True,
            "analysis": {
                "sr": 44100,
                "n_fft": 2048,
                "hop": 512,
                "bands": [[20, 160], [160, 2000], [2000, 16000]],
                "beat_track": True,
            },
        },
    }
    tmp = tmp_path / "legacy.json"
    tmp.write_text(json.dumps(legacy, indent=2), encoding="utf-8")

    model = PresetManager.load(str(tmp))
    assert model.schema_version == 1
    assert model.background.colors and len(model.background.colors) >= 2
