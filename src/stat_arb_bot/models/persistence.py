"""Versioned, safe JSON persistence for inspectable structural results."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from stat_arb_bot.models.structural import MODEL_NAME, MODEL_SCHEMA_VERSION, StructuralFitResult


def _safe_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, timedelta):
        return {"seconds": value.total_seconds()}
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _safe_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _safe_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported model persistence value: {type(value).__name__}")


class StructuralModelStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def path_for(self, fit_id: str) -> Path:
        return self.root / MODEL_NAME / f"{fit_id}.json"

    def save(self, fit: StructuralFitResult) -> Path:
        path = self.path_for(fit.metadata.fit_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": int(MODEL_SCHEMA_VERSION),
            "model_name": MODEL_NAME,
            "fit": _safe_value(fit),
        }
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return path

    def load_payload(self, fit_id: str) -> Mapping[str, Any]:
        with self.path_for(fit_id).open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("schema_version") != int(MODEL_SCHEMA_VERSION):
            raise ValueError("unsupported structural model schema version")
        if payload.get("model_name") != MODEL_NAME:
            raise ValueError("persisted model name does not match structural model")
        return payload
