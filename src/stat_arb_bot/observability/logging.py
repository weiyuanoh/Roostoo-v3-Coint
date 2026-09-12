"""Explicit structured logging configuration."""

from __future__ import annotations

import json
import logging
import logging.handlers
from datetime import datetime, timezone
from pathlib import Path


class JsonFormatter(logging.Formatter):
    """Format log records as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0]:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, sort_keys=True)


def configure_logging(
    *,
    level: str = "INFO",
    console: bool = True,
    log_dir: Path | None = None,
    filename: str = "bot.jsonl",
) -> None:
    """Configure the package logger; imports remain filesystem-side-effect free."""

    package_logger = logging.getLogger("stat_arb_bot")
    for existing_handler in package_logger.handlers[:]:
        package_logger.removeHandler(existing_handler)
        existing_handler.close()
    package_logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    package_logger.propagate = False

    if console:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        package_logger.addHandler(handler)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.TimedRotatingFileHandler(
            log_dir / filename,
            when="midnight",
            backupCount=30,
            utc=True,
        )
        file_handler.setFormatter(JsonFormatter())
        package_logger.addHandler(file_handler)

    if not package_logger.handlers:
        package_logger.addHandler(logging.NullHandler())


def get_logger(name: str) -> logging.Logger:
    """Return a child logger without configuring handlers or touching disk."""

    return logging.getLogger(f"stat_arb_bot.{name}")
