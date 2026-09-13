"""Atomic, versioned JSON checkpoints for exact sequential-filter restarts."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from stat_arb_bot.domain.execution import utc_datetime
from stat_arb_bot.models.numeric import readonly_array
from stat_arb_bot.models.state_space.config import (
    KalmanConfig,
    MeasurementNoiseConfig,
    MeasurementNoiseMode,
)
from stat_arb_bot.models.state_space.filter import (
    FILTER_MODEL_NAME,
    FILTER_MODEL_VERSION,
    FilterCheckpoint,
    FilterStatus,
)
from stat_arb_bot.models.state_space.regime import RegimeCheckpoint
from stat_arb_bot.models.state_space.representation import StateSpaceStatus


FILTER_CHECKPOINT_SCHEMA_VERSION = 1


def _measurement_payload(config: MeasurementNoiseConfig) -> dict[str, Any]:
    return {
        "mode": config.mode.value,
        "residual_variance_fraction": config.residual_variance_fraction,
        "diagonal_variances": config.diagonal_variances,
        "covariance": config.covariance,
    }


def _config_payload(config: KalmanConfig) -> dict[str, Any]:
    return {
        "measurement_noise": _measurement_payload(config.measurement_noise),
        "initialization_method": config.initialization_method,
        "initial_covariance_multiplier": config.initial_covariance_multiplier,
        "maximum_forecast_steps": config.maximum_forecast_steps,
        "standard_forecast_steps": config.standard_forecast_steps,
        "bar_minutes": config.bar_minutes,
        "psd_tolerance": config.psd_tolerance,
        "symmetry_tolerance": config.symmetry_tolerance,
        "maximum_innovation_condition_number": config.maximum_innovation_condition_number,
        "maximum_state_norm": config.maximum_state_norm,
    }


def _config_from_payload(payload: dict[str, Any]) -> KalmanConfig:
    measurement = payload["measurement_noise"]
    noise = MeasurementNoiseConfig(
        mode=MeasurementNoiseMode(measurement["mode"]),
        residual_variance_fraction=float(measurement["residual_variance_fraction"]),
        diagonal_variances=(
            tuple(float(value) for value in measurement["diagonal_variances"])
            if measurement["diagonal_variances"] is not None
            else None
        ),
        covariance=(
            tuple(tuple(float(value) for value in row) for row in measurement["covariance"])
            if measurement["covariance"] is not None
            else None
        ),
    )
    return KalmanConfig(
        measurement_noise=noise,
        initialization_method=str(payload["initialization_method"]),
        initial_covariance_multiplier=float(payload["initial_covariance_multiplier"]),
        maximum_forecast_steps=int(payload["maximum_forecast_steps"]),
        standard_forecast_steps=tuple(int(value) for value in payload["standard_forecast_steps"]),
        bar_minutes=int(payload["bar_minutes"]),
        psd_tolerance=float(payload["psd_tolerance"]),
        symmetry_tolerance=float(payload["symmetry_tolerance"]),
        maximum_innovation_condition_number=float(payload["maximum_innovation_condition_number"]),
        maximum_state_norm=float(payload["maximum_state_norm"]),
    )


def _filter_payload(checkpoint: FilterCheckpoint) -> dict[str, Any]:
    return {
        "timestamp": checkpoint.timestamp.isoformat(),
        "structural_fit_id": checkpoint.structural_fit_id,
        "structural_fit_data_end": checkpoint.structural_fit_data_end.isoformat(),
        "rank": checkpoint.rank,
        "filter_model": checkpoint.filter_model,
        "filter_version": checkpoint.filter_version,
        "state_layout_version": checkpoint.state_layout_version,
        "status": checkpoint.status.value,
        "state_mean": checkpoint.state_mean.tolist(),
        "state_covariance": checkpoint.state_covariance.tolist(),
        "last_observation": checkpoint.last_observation.tolist(),
        "initialization_method": checkpoint.initialization_method,
        "measurement_noise_covariance": checkpoint.measurement_noise_covariance.tolist(),
        "config": _config_payload(checkpoint.config),
    }


def _filter_from_payload(payload: dict[str, Any]) -> FilterCheckpoint:
    return FilterCheckpoint(
        timestamp=utc_datetime(datetime.fromisoformat(payload["timestamp"])),
        structural_fit_id=str(payload["structural_fit_id"]),
        structural_fit_data_end=utc_datetime(
            datetime.fromisoformat(payload["structural_fit_data_end"])
        ),
        rank=int(payload["rank"]),
        filter_model=str(payload["filter_model"]),
        filter_version=str(payload["filter_version"]),
        state_layout_version=str(payload["state_layout_version"]),
        status=FilterStatus(payload["status"]),
        state_mean=readonly_array(payload["state_mean"], dimensions=1),
        state_covariance=readonly_array(payload["state_covariance"], dimensions=2),
        last_observation=readonly_array(payload["last_observation"], dimensions=1),
        initialization_method=str(payload["initialization_method"]),
        measurement_noise_covariance=readonly_array(
            payload["measurement_noise_covariance"],
            dimensions=2,
        ),
        config=_config_from_payload(payload["config"]),
    )


class FilterStateStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def path_for(self, structural_fit_id: str) -> Path:
        return self.root / FILTER_MODEL_NAME / f"{structural_fit_id}.json"

    def save(self, checkpoint: FilterCheckpoint) -> Path:
        path = self.path_for(checkpoint.structural_fit_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": FILTER_CHECKPOINT_SCHEMA_VERSION,
            "filter_model": FILTER_MODEL_NAME,
            "filter_version": FILTER_MODEL_VERSION,
            "checkpoint": _filter_payload(checkpoint),
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

    def load(self, structural_fit_id: str) -> FilterCheckpoint:
        with self.path_for(structural_fit_id).open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("schema_version") != FILTER_CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("unsupported filter checkpoint schema version")
        if payload.get("filter_model") != FILTER_MODEL_NAME:
            raise ValueError("checkpoint belongs to a different filter model")
        if payload.get("filter_version") != FILTER_MODEL_VERSION:
            raise ValueError("checkpoint filter version is incompatible")
        return _filter_from_payload(payload["checkpoint"])


class RegimeStateStore:
    """Persist both active rank-one and inactive rank-regime state."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def path_for(self, structural_fit_id: str) -> Path:
        return self.root / f"{FILTER_MODEL_NAME}_regimes" / f"{structural_fit_id}.json"

    def save(self, checkpoint: RegimeCheckpoint) -> Path:
        path = self.path_for(checkpoint.structural_fit_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": FILTER_CHECKPOINT_SCHEMA_VERSION,
            "filter_model": FILTER_MODEL_NAME,
            "regime": {
                "timestamp": checkpoint.timestamp.isoformat(),
                "structural_fit_id": checkpoint.structural_fit_id,
                "rank": checkpoint.rank,
                "status": checkpoint.status.value,
                "reason": checkpoint.reason,
                "active_filter_checkpoint": (
                    _filter_payload(checkpoint.active_filter_checkpoint)
                    if checkpoint.active_filter_checkpoint
                    else None
                ),
                "final_active_checkpoint": (
                    _filter_payload(checkpoint.final_active_checkpoint)
                    if checkpoint.final_active_checkpoint
                    else None
                ),
                "config": _config_payload(checkpoint.config),
            },
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

    def load(self, structural_fit_id: str) -> RegimeCheckpoint:
        with self.path_for(structural_fit_id).open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("schema_version") != FILTER_CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("unsupported regime checkpoint schema version")
        if payload.get("filter_model") != FILTER_MODEL_NAME:
            raise ValueError("regime checkpoint belongs to a different filter model")
        value = payload["regime"]
        active = value["active_filter_checkpoint"]
        final = value["final_active_checkpoint"]
        return RegimeCheckpoint(
            timestamp=utc_datetime(datetime.fromisoformat(value["timestamp"])),
            structural_fit_id=str(value["structural_fit_id"]),
            rank=int(value["rank"]),
            status=StateSpaceStatus(value["status"]),
            reason=value["reason"],
            active_filter_checkpoint=_filter_from_payload(active) if active else None,
            final_active_checkpoint=_filter_from_payload(final) if final else None,
            config=_config_from_payload(value["config"]),
        )
