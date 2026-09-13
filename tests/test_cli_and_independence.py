from __future__ import annotations

from pathlib import Path

from stat_arb_bot.cli import _epoch_ms, _utc_timestamp, build_parser


def test_cli_exposes_no_mutating_exchange_commands() -> None:
    parser = build_parser()
    subcommands = next(
        action.choices
        for action in parser._actions  # noqa: SLF001 - parser contract inspection
        if hasattr(action, "choices") and isinstance(action.choices, dict)
    )

    assert {"place-order", "cancel-order", "open-short", "close-short"}.isdisjoint(subcommands)
    assert {
        "short-positions",
        "collect",
        "smoke",
        "research-fetch",
        "research-build",
        "research-coverage",
    }.issubset(subcommands)


def test_cli_timestamp_parser_requires_timezone() -> None:
    assert _epoch_ms("1970-01-01T00:00:01Z") == 1000
    assert _utc_timestamp("1970-01-01T00:00:01Z").isoformat() == "1970-01-01T00:00:01+00:00"


def test_source_has_no_runtime_dependency_on_v2() -> None:
    source_root = Path(__file__).resolve().parents[1] / "src"
    source = "\n".join(path.read_text(encoding="utf-8") for path in source_root.rglob("*.py"))

    assert "Roostoo-bot-v2" not in source
    assert "from bot" not in source
    assert "import bot" not in source
