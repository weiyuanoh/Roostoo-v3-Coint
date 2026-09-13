from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
from numpy.typing import NDArray

from stat_arb_bot.domain import Candle
from stat_arb_bot.research_data import ResearchUniverse, build_synchronized_panel
from stat_arb_bot.research_data.pipeline import ResearchDataBuilder, ResearchDataset
from stat_arb_bot.research_data.windows import ResearchWindow

MODEL_BASE = datetime(2024, 1, 1, tzinfo=timezone.utc)
MODEL_UNIVERSE = ResearchUniverse("USDT")


def independent_random_walks(
    observations: int = 1200,
    *,
    seed: int = 8128,
) -> NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    innovations = rng.normal(0.0, 0.006, size=(observations, 5))
    return np.asarray(np.log([50000.0, 3000.0, 120.0, 0.6, 0.4]) + np.cumsum(innovations, axis=0))


def simulate_rank_one_vecm(
    observations: int = 1400,
    *,
    seed: int = 1729,
    beta: NDArray[np.float64] | None = None,
    alpha: NDArray[np.float64] | None = None,
    gamma: float = 0.08,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    relation = np.asarray(
        beta if beta is not None else [1.0, -1.0, 0.5, -0.3, -0.2],
        dtype=np.float64,
    )
    adjustment = np.asarray(
        alpha if alpha is not None else -0.16 * relation / float(relation @ relation),
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    levels = np.empty((observations, 5), dtype=np.float64)
    levels[0] = np.log([50000.0, 3000.0, 120.0, 0.6, 0.4])
    # Start on the relation without changing the stochastic dynamics.
    levels[0, 0] -= float(relation @ levels[0]) / relation[0]
    previous_change = np.zeros(5, dtype=np.float64)
    for index in range(1, observations):
        equilibrium = float(relation @ levels[index - 1])
        innovation = rng.normal(0.0, 0.006, size=5)
        change = adjustment * equilibrium + gamma * previous_change + innovation
        levels[index] = levels[index - 1] + change
        previous_change = change
    return levels, relation, adjustment


def one_common_trend_system(
    observations: int = 1200,
    *,
    seed: int = 440,
) -> NDArray[np.float64]:
    """Five levels with one common trend and four stationary combinations."""

    rng = np.random.default_rng(seed)
    trend = np.cumsum(rng.normal(0.0, 0.006, observations))
    levels = np.empty((observations, 5), dtype=np.float64)
    for asset_index in range(5):
        stationary = np.empty(observations)
        stationary[0] = 0.0
        shocks = rng.normal(0.0, 0.002, observations)
        for index in range(1, observations):
            stationary[index] = 0.55 * stationary[index - 1] + shocks[index]
        levels[:, asset_index] = 4.0 + asset_index * 0.2 + trend + stationary
    return levels


def research_window_from_levels(
    levels: NDArray[np.float64],
    *,
    end_index: int | None = None,
    minimum_observations: int = 50,
) -> ResearchWindow:
    rows = len(levels) if end_index is None else end_index + 1
    candles: list[Candle] = []
    retrieval = MODEL_BASE + timedelta(minutes=15 * (len(levels) + 1))
    for index, values in enumerate(levels):
        open_time = MODEL_BASE + timedelta(minutes=15 * index)
        for asset, value in zip(MODEL_UNIVERSE.assets, values):
            price = float(np.exp(value))
            candles.append(
                Candle(
                    open_time=open_time,
                    close_time=open_time + timedelta(minutes=15),
                    symbol=MODEL_UNIVERSE.pair(asset),
                    interval="15m",
                    open=price,
                    high=price * 1.001,
                    low=price * 0.999,
                    close=price,
                    volume=100.0,
                    source="SYNTHETIC",
                    retrieved_at=retrieval,
                )
            )
    end = MODEL_BASE + timedelta(minutes=15 * rows)
    panel = build_synchronized_panel(candles, universe=MODEL_UNIVERSE, as_of=end)
    return panel.window(
        end=end,
        observations=rows,
        minimum_observations=minimum_observations,
    )


def research_dataset_from_levels(levels: NDArray[np.float64]) -> ResearchDataset:
    raw_by_asset: dict[str, list[Candle]] = {asset: [] for asset in MODEL_UNIVERSE.assets}
    retrieval = MODEL_BASE + timedelta(minutes=15 * (len(levels) + 1))
    for model_index, values in enumerate(levels):
        for constituent in range(3):
            raw_index = model_index * 3 + constituent
            open_time = MODEL_BASE + timedelta(minutes=5 * raw_index)
            for asset, value in zip(MODEL_UNIVERSE.assets, values):
                price = float(np.exp(value))
                raw_by_asset[asset].append(
                    Candle(
                        open_time=open_time,
                        close_time=open_time + timedelta(minutes=5) - timedelta(milliseconds=1),
                        symbol=MODEL_UNIVERSE.pair(asset),
                        interval="5m",
                        open=price,
                        high=price * 1.001,
                        low=price * 0.999,
                        close=price,
                        volume=20.0,
                        source="SYNTHETIC",
                        retrieved_at=retrieval,
                    )
                )
    end = MODEL_BASE + timedelta(minutes=15 * len(levels))
    return ResearchDataBuilder(MODEL_UNIVERSE).build(
        raw_by_asset,
        as_of=end,
        requested_start=MODEL_BASE,
    )
