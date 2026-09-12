"""Roostoo HMAC-SHA256 authentication helpers."""

from __future__ import annotations

import hashlib
import hmac
import time
from collections.abc import Mapping
from typing import Any


def timestamp_ms() -> str:
    return str(int(time.time() * 1000))


def encode_params(params: Mapping[str, Any]) -> str:
    """Encode parameters in the sorted, unescaped form required by Roostoo."""

    return "&".join(f"{key}={params[key]}" for key in sorted(params))


def sign_params(
    params: Mapping[str, Any],
    api_secret: str,
    *,
    now_ms: str | None = None,
) -> tuple[dict[str, str], str, dict[str, Any]]:
    """Return signature headers, encoded body/query, and timestamped payload."""

    signed = {**params, "timestamp": now_ms or timestamp_ms()}
    encoded = encode_params(signed)
    signature = hmac.new(
        api_secret.encode("utf-8"),
        encoded.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return {"MSG-SIGNATURE": signature}, encoded, signed
