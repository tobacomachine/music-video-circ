import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import os


LOG_DIR = Path(os.environ.get("USERPROFILE", Path.home())) / ".ncsviz" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def configure_logging() -> None:
    """Configure application logging with rotation."""
    handler = RotatingFileHandler(LOG_DIR / "app.log", maxBytes=10 * 1024 * 1024, backupCount=5)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        handlers=[handler],
    )
