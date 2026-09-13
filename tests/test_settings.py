from __future__ import annotations

import logging
from pathlib import Path

import pytest

from stat_arb_bot.config import SettingsError, load_settings
from stat_arb_bot.observability.logging import configure_logging, get_logger


def test_settings_defaults_are_resolved_against_project_root(tmp_path: Path) -> None:
    settings = load_settings(environ={}, project_root=tmp_path)

    assert settings.project_root == tmp_path.resolve()
    assert settings.data_dir == tmp_path / "data/candles"
    assert settings.research_data_dir == tmp_path / "data/research"
    assert settings.research_quote_currency == "USDT"
    assert settings.log_dir == tmp_path / "logs"
    assert settings.binance_base_urls == (
        "https://data-api.binance.vision",
        "https://api.binance.com",
    )


def test_process_environment_overrides_dotenv(tmp_path: Path) -> None:
    env_file = tmp_path / "test.env"
    env_file.write_text(
        "ROOSTOO_API_KEY=file-key\nLOG_LEVEL=WARNING\nBOT_LOG_FILE=1\n",
        encoding="utf-8",
    )

    settings = load_settings(
        env_file,
        environ={"ROOSTOO_API_KEY": "process-key", "BOT_LOG_FILE": "0"},
        project_root=tmp_path,
    )

    assert settings.roostoo_api_key == "process-key"
    assert settings.log_level == "WARNING"
    assert settings.log_file is False


def test_research_quote_and_storage_root_are_configurable(tmp_path: Path) -> None:
    settings = load_settings(
        environ={
            "RESEARCH_QUOTE_CURRENCY": "usdc",
            "RESEARCH_DATA_DIR": "datasets/research",
        },
        project_root=tmp_path,
    )

    assert settings.research_quote_currency == "USDC"
    assert settings.research_data_dir == tmp_path / "datasets/research"


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("REQUEST_TIMEOUT_SECONDS", "0"),
        ("HTTP_MAX_ATTEMPTS", "nope"),
        ("HTTP_BACKOFF_SECONDS", "-1"),
        ("BOT_LOG_CONSOLE", "sometimes"),
        ("LOG_LEVEL", "VERBOSE"),
        ("ROOSTOO_BASE_URL", "not-a-url"),
        ("BINANCE_FALLBACK_URLS", "ftp://invalid.test"),
        ("RESEARCH_QUOTE_CURRENCY", "USD/T"),
    ],
)
def test_invalid_settings_are_rejected(tmp_path: Path, name: str, value: str) -> None:
    with pytest.raises(SettingsError):
        load_settings(environ={name: value}, project_root=tmp_path)


def test_credentials_are_required_only_when_requested(tmp_path: Path) -> None:
    settings = load_settings(environ={}, project_root=tmp_path)

    with pytest.raises(SettingsError, match="ROOSTOO_API_KEY"):
        settings.require_roostoo_credentials()


def test_get_logger_has_no_filesystem_side_effect(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    logger = get_logger("test")

    assert isinstance(logger, logging.Logger)
    assert not log_dir.exists()

    configure_logging(console=False, log_dir=log_dir)
    logger.info("configured explicitly")
    assert (log_dir / "bot.jsonl").exists()
