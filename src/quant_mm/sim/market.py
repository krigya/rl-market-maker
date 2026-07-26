"""The latent 'fair value' process and market impact.

Real prices have a component you can't see directly: the **fundamental** (or
efficient) price -- where the asset 'should' trade given all information. The
observable order book wiggles around it. We model the fundamental as an
**arithmetic Brownian motion** (the same assumption Avellaneda-Stoikov make, so
our baseline is exact rather than approximate):

    dS = sigma * dW          (no drift -- a fair game)

Two consequences the market maker must live with flow from this object:

* **Inventory risk** -- once you hold inventory, fundamental moves are P&L you
  did not choose to take.
* **Market impact** -- aggressive order flow does not just *reveal* the
  fundamental, it *pushes* it. We add a linear permanent-impact term so net
  buying nudges the fundamental up. This is the (partial) sense in which 'your
  own quotes move the market': trades that lift your offers shift fair value.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class FundamentalConfig:
    s0: float = 100.0       # starting fair value
    sigma: float = 0.01     # per-step stdev of fundamental increments (price units)
    impact: float = 1e-4    # permanent price impact per unit of net signed volume


class Fundamental:
    """An arithmetic-Brownian-motion fair value with linear permanent impact."""

    def __init__(self, cfg: FundamentalConfig, rng: np.random.Generator) -> None:
        self.cfg = cfg
        self.rng = rng
        self.value = cfg.s0

    def reset(self) -> None:
        self.value = self.cfg.s0

    def planned_increment(self) -> float:
        """Draw this step's *exogenous* fundamental move (before impact).

        Returned separately from :meth:`apply` so the simulator can make informed
        order flow correlate with where the price is about to go -- the mechanism
        that produces **adverse selection** against the market maker.
        """
        return float(self.rng.normal(0.0, self.cfg.sigma))

    def apply(self, exogenous_increment: float, net_signed_volume: float = 0.0) -> None:
        """Advance the fair value by the exogenous move plus market impact."""
        self.value += exogenous_increment + self.cfg.impact * net_signed_volume
