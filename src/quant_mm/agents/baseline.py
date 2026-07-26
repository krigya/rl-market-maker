"""Analytic and heuristic quoting policies - the agents the RL must beat.

The headline is :class:`AvellanedaStoikovAgent`, the closed-form optimal market
maker from Avellaneda & Stoikov (2008). It is the natural benchmark: a
principled, inventory-aware quoter derived from stochastic optimal control, with
*no learning*. If a trained DQN/PPO can't beat it, the RL hasn't earned its keep.

The Avellaneda-Stoikov rule
---------------------------
Around the observable mid ``s``, with inventory ``q`` and time-to-horizon
``(T - t)``, define a **reservation price** - the inventory-adjusted fair value
the maker quotes around:

    r(s, t) = s - q * gamma * sigma^2 * (T - t)

When the maker is long (``q > 0``) the reservation price sits *below* mid, so
both quotes shift down: it becomes keener to sell and shyer to buy, bleeding off
the position. That asymmetry *is* inventory management. The optimal total spread
balances earning the spread against the risk of holding inventory and the order
arrival intensity ``lambda(d) = A * exp(-kappa * d)``:

    delta = gamma * sigma^2 * (T - t) + (2 / gamma) * ln(1 + gamma / kappa)

giving quotes ``bid = r - delta/2`` and ``ask = r + delta/2``.

All agents here expose the same ``act(obs)`` interface as the RL agents, so the
evaluation harness can treat them interchangeably.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..sim.env import MMEnvConfig


def offsets_to_action(bid_ticks: int, ask_ticks: int, n_offsets: int) -> int:
    """Map integer tick offsets (1..n_offsets) to the env's discrete action id."""
    b = int(np.clip(bid_ticks, 1, n_offsets)) - 1
    a = int(np.clip(ask_ticks, 1, n_offsets)) - 1
    return b * n_offsets + a


@dataclass
class ASConfig:
    """Risk aversion and order-arrival decay for Avellaneda-Stoikov.

    ``kappa`` is expressed in price units; it should track the empirical fill
    intensity decay measured from the environment (see 02_baseline_as.ipynb).
    """

    gamma: float = 0.1     # inventory risk aversion
    kappa: float = 40.0    # order-arrival intensity decay (1/price)


class AvellanedaStoikovAgent:
    """Closed-form inventory-aware market maker, projected onto the tick grid."""

    def __init__(self, env_cfg: MMEnvConfig, as_cfg: ASConfig | None = None) -> None:
        self.env = env_cfg
        self.cfg = as_cfg or ASConfig()

    def act(self, obs: np.ndarray) -> int:
        # Recover raw quantities from the normalised observation.
        q = float(obs[0]) * self.env.max_inventory          # signed inventory
        time_remaining = float(obs[3]) * self.env.horizon   # (T - t), in steps
        sigma2 = self.env.sigma**2
        g, k = self.cfg.gamma, self.cfg.kappa

        # Reservation-price skew and optimal half-spread (in price units).
        skew = q * g * sigma2 * time_remaining
        half_spread = 0.5 * (g * sigma2 * time_remaining + (2.0 / g) * math.log1p(g / k))

        # bid/ask distance from mid; long inventory -> wider bid, tighter ask.
        bid_offset = (skew + half_spread) / self.env.tick     # in ticks
        ask_offset = (-skew + half_spread) / self.env.tick

        return offsets_to_action(
            round(bid_offset), round(ask_offset), self.env.n_offsets
        )

    def reset(self) -> None:  # stateless; kept for a uniform interface
        pass


class FixedSymmetricAgent:
    """Quote a constant offset on both sides - the simplest naive market maker."""

    def __init__(self, env_cfg: MMEnvConfig, offset_ticks: int = 3) -> None:
        self.n = env_cfg.n_offsets
        self.offset = offset_ticks

    def act(self, obs: np.ndarray) -> int:
        return offsets_to_action(self.offset, self.offset, self.n)

    def reset(self) -> None:
        pass


class RandomAgent:
    """Uniformly random actions - a sanity floor for the comparison."""

    def __init__(self, env_cfg: MMEnvConfig, seed: int | None = None) -> None:
        self.n_actions = env_cfg.n_offsets**2
        self.rng = np.random.default_rng(seed)

    def act(self, obs: np.ndarray) -> int:
        return int(self.rng.integers(self.n_actions))

    def reset(self) -> None:
        pass
