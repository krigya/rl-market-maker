"""Historical price replay for out-of-distribution validation.

The agents were trained on a fundamental whose increments are i.i.d. Gaussian.
Real markets are not: returns are fat-tailed, autocorrelated in magnitude
(volatility clustering), and occasionally trend. This module loads a real price
series and turns it into fundamental increments the environment can replay, so
we can ask the honest question: *does a policy trained on clean synthetic data
survive a realistic return distribution?*

We standardise the increments to the training volatility (``target_sigma``) so
that **scale** is held fixed and only the **distributional shape** changes --
isolating the sim-to-real gap that matters for the learned quote offsets. The
loader accepts any CSV with a close-price column, so a LOBSTER mid-price series
or a crypto tape drops straight in.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

# Column names we will treat as the price series, in order of preference.
_CLOSE_CANDIDATES = ("Close", "close", "AAPL.Close", "Adj Close", "price", "mid", "Mid")


def load_price_series(path: str | pathlib.Path, column: str | None = None) -> np.ndarray:
    """Load a 1-D price series from a CSV.

    If ``column`` is not given, the first recognised close-like column is used;
    failing that, the last numeric column.
    """
    df = pd.read_csv(path)
    if column is None:
        column = next((c for c in _CLOSE_CANDIDATES if c in df.columns), None)
        if column is None:  # fall back to the last numeric column
            numeric = df.select_dtypes("number").columns
            if len(numeric) == 0:
                raise ValueError(f"no numeric price column found in {path}")
            column = numeric[-1]
    return df[column].to_numpy(dtype=np.float64)


def replay_increments(prices: np.ndarray, target_sigma: float) -> np.ndarray:
    """Convert prices to per-step increments rescaled to ``target_sigma``.

    The *shape* of the return distribution (fat tails, volatility clustering) is
    preserved; only the standard deviation is matched to the training regime.
    """
    diffs = np.diff(np.asarray(prices, dtype=np.float64))
    diffs = diffs[np.isfinite(diffs)]
    std = diffs.std()
    if std < 1e-12:
        raise ValueError("price series has (near-)zero volatility")
    return diffs * (target_sigma / std)


def distribution_stats(increments: np.ndarray) -> dict[str, float]:
    """Summary stats highlighting how non-Gaussian the series is."""
    x = np.asarray(increments, dtype=np.float64)
    x = x[np.isfinite(x)]
    mean, std = x.mean(), x.std()
    z = (x - mean) / (std + 1e-12)
    skew = float(np.mean(z**3))
    excess_kurtosis = float(np.mean(z**4) - 3.0)  # 0 for a Gaussian
    # lag-1 autocorrelation of |returns| - the volatility-clustering signature.
    a = np.abs(x - mean)
    vol_autocorr = float(np.corrcoef(a[:-1], a[1:])[0, 1]) if len(a) > 2 else float("nan")
    return {
        "n": int(len(x)),
        "std": float(std),
        "skew": skew,
        "excess_kurtosis": excess_kurtosis,
        "abs_return_autocorr": vol_autocorr,
    }
