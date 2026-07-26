"""Background order-flow arrival processes.

The market maker quotes into a market populated by *other people's* orders. The
rate at which marketable (liquidity-taking) orders arrive is what gives the MM
its fills. We offer two models of that arrival rate:

* :class:`PoissonArrivals` -- the textbook assumption: arrivals are independent,
  constant-rate. Simple, and exactly the assumption behind Avellaneda-Stoikov.

* :class:`HawkesArrivals` -- a **self-exciting** process: each arrival
  temporarily raises the chance of more arrivals, so trades cluster in bursts.
  This is what real order flow actually looks like, and it injects the
  **non-stationarity** the RL agent has to cope with but the static AS formula
  ignores. Implemented in discrete time with an exponential kernel:

      lambda_t = mu + (lambda_{t-1} - mu) * exp(-beta) + alpha * N_{t-1}

  i.e. intensity decays geometrically toward baseline ``mu`` and jumps by
  ``alpha`` for every order that just arrived.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class PoissonArrivals:
    """Constant-rate arrivals: ``N_t ~ Poisson(rate)`` each step."""

    rate: float = 1.0

    def reset(self) -> None:  # nothing to reset; here for a uniform interface
        pass

    def sample(self, rng: np.random.Generator) -> int:
        return int(rng.poisson(self.rate))


@dataclass(slots=True)
class HawkesArrivals:
    """Self-exciting arrivals with an exponential decay kernel.

    Parameters
    ----------
    mu:    baseline intensity (arrivals/step with no recent activity).
    alpha: intensity added per arrival (clustering strength).
    beta:  decay rate of excitement (larger = forgets faster).

    Stationarity requires ``alpha / (1 - exp(-beta)) < 1`` (the branching ratio);
    we expose :attr:`branching_ratio` so callers can check.
    """

    mu: float = 0.6
    alpha: float = 0.3
    beta: float = 1.0
    intensity: float = 0.0

    def __post_init__(self) -> None:
        self.intensity = self.mu

    @property
    def branching_ratio(self) -> float:
        return self.alpha / (1.0 - math.exp(-self.beta))

    def reset(self) -> None:
        self.intensity = self.mu

    def sample(self, rng: np.random.Generator) -> int:
        # 1. decay yesterday's excitement toward baseline.
        self.intensity = self.mu + (self.intensity - self.mu) * math.exp(-self.beta)
        # 2. draw arrivals at the current intensity.
        n = int(rng.poisson(self.intensity))
        # 3. each arrival excites future intensity.
        self.intensity += self.alpha * n
        return n
