from pathlib import Path
import json
import sys

# ensure repository root on path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from app.core.preset_manager import PresetManager, PresetModel


def test_minimal_ring_valid():
    path = Path('app/assets/presets/minimal_ring.json')
    model = PresetManager.load(path)
    assert isinstance(model, PresetModel)
    assert model.visual.mode.ring is True
    assert model.visual.mode.bars is False


def test_round_trip(tmp_path):
    path = Path('app/assets/presets/minimal_ring.json')
    model = PresetManager.load(path)
    out = tmp_path / 'out.json'
    PresetManager.save(model, out)
    model2 = PresetManager.load(out)
    d1 = model.model_dump()
    d2 = model2.model_dump()
    d1['meta']['created'] = d2['meta']['created'] = 'x'
    assert d1 == d2


def test_migration(tmp_path):
    raw = {
        "meta": {"name": "legacy"},
        "bg": {"type": "solid", "colors": ["rgb(255,0,0)"]},
        "vis": {"mode": {"ring": True, "bars": False}}
    }
    p = tmp_path / 'legacy.json'
    p.write_text(json.dumps(raw), encoding='utf-8')
    model = PresetManager.load(p)
    assert model.schema_version == 1
    assert model.background.type == 'solid'
    assert model.background.colors[0] == '#ff0000'
    assert model.visual.mode.bars is False
