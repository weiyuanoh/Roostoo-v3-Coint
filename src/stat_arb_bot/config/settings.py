"""Validated, side-effect-free runtime settings.

Unlike the V2 configuration module, importing this module never reads a file,
creates a directory, or opens a log. Call :func:`load_settings` explicitly from
an application entry point.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from dotenv import dotenv_values


class SettingsError(ValueError):
    """Raised when runtime configuration is invalid."""


def _csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip().rstrip("/") for item in value.split(",") if item.strip())


def _url(value: str, *, name: str) -> str:
    normalized = value.strip().rstrip("/")
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise SettingsError(f"{name} must be an HTTP(S) URL")
    return normalized


def _bool(value: str, *, name: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise SettingsError(f"{name} must be a boolean value")


def _positive_float(value: str, *, name: str, allow_zero: bool = False) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise SettingsError(f"{name} must be numeric") from exc
    invalid = parsed < 0 if allow_zero else parsed <= 0
    if invalid:
        qualifier = "non-negative" if allow_zero else "positive"
        raise SettingsError(f"{name} must be {qualifier}")
    return parsed


def _positive_int(value: str, *, name: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise SettingsError(f"{name} must be an integer") from exc
    if parsed <= 0:
        raise SettingsError(f"{name} must be positive")
    return parsed


@dataclass(frozen=True)
class Settings:
    """Infrastructure-only settings for exchange and market-data access."""

    project_root: Path
    roostoo_api_key: str = ""
    roostoo_api_secret: str = ""
    roostoo_base_url: str = "https://mock-api.roostoo.com"
    binance_base_urls: tuple[str, ...] = (
        "https://data-api.binance.vision",
        "https://api.binance.com",
    )
    request_timeout_seconds: float = 10.0
    http_max_attempts: int = 3
    http_backoff_seconds: float = 0.25
    data_dir: Path = Path("data/candles")
    log_dir: Path = Path("logs")
    log_level: str = "INFO"
    log_console: bool = True
    log_file: bool = True

    def require_roostoo_credentials(self) -> None:
        if not self.roostoo_api_key or not self.roostoo_api_secret:
            raise SettingsError(
                "ROOSTOO_API_KEY and ROOSTOO_API_SECRET are required for signed endpoints"
            )


def load_settings(
    env_file: str | Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    project_root: str | Path | None = None,
) -> Settings:
    """Load settings explicitly from an optional dotenv file and the environment.

    Process environment values take precedence over the dotenv file. Relative
    data and log paths are resolved against ``project_root``.
    """

    root = Path(project_root or Path(__file__).resolve().parents[3]).resolve()
    values: dict[str, str] = {}
    if env_file is not None:
        file_path = Path(env_file)
        if not file_path.is_absolute():
            file_path = root / file_path
        if not file_path.exists():
            raise SettingsError(f"environment file does not exist: {file_path}")
        values.update(
            {
                str(key): str(value)
                for key, value in dotenv_values(file_path).items()
                if value is not None
            }
        )
    values.update(dict(os.environ if environ is None else environ))

    def get(name: str, default: str) -> str:
        return values.get(name, default)

    primary = _url(
        get("BINANCE_BASE_URL", "https://data-api.binance.vision"),
        name="BINANCE_BASE_URL",
    )
    fallbacks = _csv(
        get(
            "BINANCE_FALLBACK_URLS",
            "https://data-api.binance.vision,https://api.binance.com",
        )
    )
    base_urls = tuple(
        dict.fromkeys(
            _url(url, name="BINANCE_FALLBACK_URLS") for url in (primary, *fallbacks) if url
        )
    )
    if not base_urls:
        raise SettingsError("at least one Binance base URL is required")

    def resolve_path(raw: str) -> Path:
        path = Path(raw)
        return path if path.is_absolute() else root / path

    level = get("LOG_LEVEL", "INFO").strip().upper()
    if level not in {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}:
        raise SettingsError("LOG_LEVEL must be CRITICAL, ERROR, WARNING, INFO, or DEBUG")

    return Settings(
        project_root=root,
        roostoo_api_key=get("ROOSTOO_API_KEY", "").strip(),
        roostoo_api_secret=get("ROOSTOO_API_SECRET", "").strip(),
        roostoo_base_url=_url(
            get("ROOSTOO_BASE_URL", "https://mock-api.roostoo.com"),
            name="ROOSTOO_BASE_URL",
        ),
        binance_base_urls=base_urls,
        request_timeout_seconds=_positive_float(
            get("REQUEST_TIMEOUT_SECONDS", "10"), name="REQUEST_TIMEOUT_SECONDS"
        ),
        http_max_attempts=_positive_int(get("HTTP_MAX_ATTEMPTS", "3"), name="HTTP_MAX_ATTEMPTS"),
        http_backoff_seconds=_positive_float(
            get("HTTP_BACKOFF_SECONDS", "0.25"),
            name="HTTP_BACKOFF_SECONDS",
            allow_zero=True,
        ),
        data_dir=resolve_path(get("DATA_DIR", "data/candles")),
        log_dir=resolve_path(get("LOG_DIR", "logs")),
        log_level=level,
        log_console=_bool(get("BOT_LOG_CONSOLE", "1"), name="BOT_LOG_CONSOLE"),
        log_file=_bool(get("BOT_LOG_FILE", "1"), name="BOT_LOG_FILE"),
    )
