"""Tests for historical price replay (Phase 5).

Self-contained: we synthesise a small CSV in a temp dir rather than depending on
any downloaded data file, so the suite runs anywhere.
"""

import numpy as np
import pandas as pd

from quant_mm.sim import (
    MMEnvConfig,
    MarketMakingEnv,
    distribution_stats,
    load_price_series,
    replay_increments,
)


def _make_csv(tmp_path, prices, col="Close"):
    path = tmp_path / "prices.csv"
    pd.DataFrame({"Date": range(len(prices)), col: prices}).to_csv(path, index=False)
    return path


def test_load_price_series_finds_close_column(tmp_path) -> None:
    prices = np.linspace(100, 110, 50)
    series = load_price_series(_make_csv(tmp_path, prices))
    assert series.shape == (50,)
    assert np.isclose(series[0], 100.0)


def test_replay_increments_match_target_sigma(tmp_path) -> None:
    rng = np.random.default_rng(0)
    prices = 100 + np.cumsum(rng.standard_normal(2000))
    inc = replay_increments(prices, target_sigma=0.02)
    assert np.isclose(inc.std(), 0.02, rtol=1e-6)


def test_distribution_stats_flags_fat_tails() -> None:
    rng = np.random.default_rng(1)
    fat = rng.standard_t(df=3, size=20000)  # heavy-tailed
    stats = distribution_stats(fat)
    assert stats["excess_kurtosis"] > 1.0  # clearly non-Gaussian


def test_env_replays_provided_increments_deterministically() -> None:
    inc = np.linspace(-0.01, 0.01, 500)
    env = MarketMakingEnv(MMEnvConfig(horizon=50, replay_increments=inc))

    def fundamental_path(seed):
        env.reset(seed=seed)
        path = []
        done = False
        while not done:
            _, _, term, trunc, info = env.step(0)
            path.append(info["fair"])
            done = term or trunc
        return path

    # Same seed -> identical replay window -> identical fundamental path.
    assert fundamental_path(0) == fundamental_path(0)


def test_replay_env_differs_from_synthetic() -> None:
    inc = np.full(500, 0.05)  # a strong constant uptrend, unlike Gaussian noise
    rep = MarketMakingEnv(MMEnvConfig(horizon=50, replay_increments=inc))
    syn = MarketMakingEnv(MMEnvConfig(horizon=50))

    def final_fair(env, seed):
        env.reset(seed=seed)
        done = False
        info = {}
        while not done:
            _, _, term, trunc, info = env.step(0)
            done = term or trunc
        return info["fair"]

    # The relentless uptrend must push fair value far above the synthetic case.
    assert final_fair(rep, 0) > final_fair(syn, 0) + 1.0
