"""Atomic JSON persistence for portfolio accounting state."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from stat_arb_bot.accounting import Portfolio


class PortfolioStateError(ValueError):
    """Raised when persisted portfolio state is missing or invalid."""


class PortfolioStateStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def save(self, portfolio: Portfolio) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            serialized = json.dumps(portfolio.to_state(), indent=2, sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise PortfolioStateError("portfolio state is not JSON serializable") from exc

        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                handle.write(serialized)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, self.path)
            temporary_path = None
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        return self.path

    def load(self) -> Portfolio:
        if not self.path.exists():
            raise PortfolioStateError(f"portfolio state does not exist: {self.path}")
        try:
            payload: Any = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise PortfolioStateError("portfolio state root must be an object")
            return Portfolio.from_state(payload)
        except PortfolioStateError:
            raise
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise PortfolioStateError(f"invalid portfolio state: {self.path}") from exc
