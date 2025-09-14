# NCS Visualizer

Base del proyecto **ncs-visualizer**. Requiere Python 3.11.

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Ejecutar la aplicación

```bash
python run.py
```

## Tests y calidad

```bash
ruff check .
pytest -q
```

## Análisis de audio (CLI)

```bash
python -m app.cli.analyze_audio --audio assets/example.wav --out analysis.json
```

## Presets

Los presets JSON se ubican en `assets/presets`. Incluye un preset de ejemplo
`minimal_ring.json` y otros placeholders.
