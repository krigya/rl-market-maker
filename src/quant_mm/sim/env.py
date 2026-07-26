"""The market-making environment (Gymnasium-compatible).

This is where everything meets: the real :class:`MatchingEngine` is the venue,
the :class:`Fundamental` drives fair value, and the arrival process feeds
background flow. The agent is a market maker that re-quotes a two-sided market
every step and tries to earn the spread without accumulating dangerous
inventory.

Why fills, impact, and adverse selection are *emergent*, not scripted
-------------------------------------------------------------------
Most RL market-making demos use a reduced-form fill model - they *assume* a
fill probability ``lambda(delta) = A*exp(-k*delta)``. We don't. Instead:

* Background liquidity providers post a ladder of limit orders each step.
* The agent's quotes rest in the **same book**, competing for queue position.
* Liquidity-taking market orders walk the book by price-time priority.

So if the agent quotes *tighter* (closer to mid) it jumps ahead of background
liquidity and fills more often but earns less; quote *wider* and background
orders shield it, so it fills rarely but earns more per fill. That tradeoff --
the heart of market making - falls out of the matching engine for free. And
because a fraction of market orders are **informed** (their direction is
correlated with the next fundamental move), the agent that gets filled is
disproportionately filled right before the price moves against it: that is
**adverse selection**, observed rather than hard-coded.

State / action / reward (the RL formulation)
--------------------------------------------
* **state**  = ``[inventory, book imbalance, spread, time-remaining,
  recent return, realised volatility]`` (all normalised).
* **action** = a discrete ``(bid_offset, ask_offset)`` pair in ticks. Asymmetric
  offsets let the agent *skew* quotes to lean against its inventory.
* **reward** = ``dWealth - inventory_penalty * q^2`` each step, with a forced
  liquidation at the horizon so unflattened inventory is paid for.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

try:  # gymnasium is the standard, but keep the env importable without it
    import gymnasium as gym
    from gymnasium import spaces
    _BASE = gym.Env
except Exception:  # pragma: no cover
    gym = None
    spaces = None
    _BASE = object

from ..lob import MatchingEngine, OrderType, Side, Trade


@dataclass
class MMEnvConfig:
    """All tunable knobs for the environment, in one reproducible place."""

    # --- price grid & horizon ---
    tick: float = 0.01
    horizon: int = 200                # decision steps per episode (the AS 'T')

    # --- fundamental process ---
    s0: float = 100.0
    sigma: float = 0.01               # per-step fundamental volatility (price units)
    impact: float = 1e-4              # permanent impact per unit net signed volume

    # --- background flow ---
    use_hawkes: bool = False          # False -> Poisson arrivals; True -> clustered
    arrival_rate: float = 1.5         # Poisson mean market orders/step
    hawkes_mu: float = 0.8
    hawkes_alpha: float = 0.3
    hawkes_beta: float = 1.0
    max_market_order_size: int = 12   # fat enough to punch through several levels
    informed_ratio: float = 0.6       # fraction of market orders that 'know' the move

    # --- background limit-order book shape ---
    # Calibrated so that the agent's fill rate decays *smoothly* across the four
    # quote offsets (verified in 02_baseline_as.ipynb): with `bg_level_size`
    # resting at each tighter level, a market order reaches the agent at offset k
    # only if it exceeds ~bg_level_size*(k-1) shares, giving the exp-like decay
    # that the Avellaneda-Stoikov intensity lambda(d)=A*exp(-k*d) assumes.
    bg_depth: int = 5                 # number of ladder levels per side
    bg_min_offset: int = 2            # tightest background quote, in ticks from mid
    bg_level_size: int = 3            # resting size posted at each background level

    # --- the market maker (agent) ---
    n_offsets: int = 4                # action grid: offsets 1..n_offsets ticks
    mm_order_size: int = 5            # size the agent quotes on each side
    max_inventory: int = 50           # hard risk limit (positions are clamped here)
    continuous_action: bool = False   # False -> Discrete (DQN); True -> Box (PPO)

    # --- reward shaping ---
    inventory_penalty: float = 1e-3   # coefficient on q^2 (the 'fight' to tune)
    vol_window: int = 20              # lookback for the realised-vol feature

    # --- historical replay (Phase 5) ---
    # If set, the fundamental's per-step increments are drawn from this real
    # series (a random window per episode) instead of Gaussian noise - an
    # out-of-distribution test on a real return distribution.
    replay_increments: "np.ndarray | None" = field(default=None, repr=False)

    @property
    def price_decimals(self) -> int:
        return max(0, int(round(-math.log10(self.tick))))


class MarketMakingEnv(_BASE):
    """A single-asset market-making environment over the real matching engine."""

    metadata = {"render_modes": []}

    def __init__(self, config: MMEnvConfig | None = None) -> None:
        super().__init__()
        self.cfg = config or MMEnvConfig()
        c = self.cfg

        if spaces is not None:
            if c.continuous_action:
                # Continuous action: 2 values in [-1, 1] -> (bid, ask) offsets,
                # mapped to [1, n_offsets] ticks. Used by the policy-gradient (PPO)
                # agent; quotes are rounded to the tick grid on placement.
                self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
            else:
                # Discrete action: index -> (bid_offset_idx, ask_offset_idx).
                self.action_space = spaces.Discrete(c.n_offsets * c.n_offsets)
            # 6 normalised features; generous bounds for float32 safety.
            self.observation_space = spaces.Box(
                low=-10.0, high=10.0, shape=(6,), dtype=np.float32
            )

        # runtime state (initialised in reset)
        self.engine: MatchingEngine
        self.np_random: np.random.Generator
        self._reset_runtime()

    # ------------------------------------------------------------------ #
    #  Gymnasium API
    # ------------------------------------------------------------------ #

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if hasattr(super(), "reset"):
            super().reset(seed=seed)
        if seed is not None or not hasattr(self, "np_random"):
            self.np_random = np.random.default_rng(seed)

        self._reset_runtime()
        self.engine = MatchingEngine()
        self.engine.subscribe(self._on_trade)

        self._fair = self.cfg.s0
        if self.cfg.use_hawkes:
            self._hawkes_intensity = self.cfg.hawkes_mu

        # Historical replay: pick a random window of real increments for this episode.
        rep = self.cfg.replay_increments
        if rep is not None and len(rep) > self.cfg.horizon:
            start = int(self.np_random.integers(0, len(rep) - self.cfg.horizon))
            self._replay = np.asarray(rep[start:start + self.cfg.horizon], dtype=np.float64)
        else:
            self._replay = None

        self._post_background(self._fair)
        self._prev_mid = self._observed_mid()
        self._prev_wealth = self._wealth()

        obs = self._build_obs(reward_return=0.0)
        return obs, self._info()

    def step(self, action: int):
        c = self.cfg

        # 1. Agent quotes a two-sided market around the current observable mid.
        ref_mid = self._observed_mid()
        bid_ticks, ask_ticks = self._decode_action(action)
        self._place_agent_quotes(ref_mid, bid_ticks, ask_ticks)

        # 2. The market moves and background flow arrives (may fill the agent).
        #    Replay a real increment if provided, else draw synthetic Gaussian noise.
        if self._replay is not None:
            planned = float(self._replay[self.t])
        else:
            planned = self.np_random.normal(0.0, c.sigma)      # exogenous move
        self._step_fills = 0
        net_signed = self._run_market_orders(planned)

        # 3. Fundamental advances (exogenous move + permanent impact of flow).
        self._fair += planned + c.impact * net_signed

        # 4. Mark to fair value and compute the shaped reward.
        wealth = self._wealth()
        reward = (wealth - self._prev_wealth) - c.inventory_penalty * self.inventory**2
        self._prev_wealth = wealth

        # 5. Observation reflects the *post-trade* (depleted) book -> imbalance
        #    carries information about the flow that just happened.
        mid_now = self._observed_mid()
        ret = (mid_now - self._prev_mid) if (mid_now and self._prev_mid) else 0.0
        self._mid_returns.append(ret)
        self._prev_mid = mid_now
        obs = self._build_obs(reward_return=ret)

        # 6. Advance the clock; refresh the book for the next step.
        self.t += 1
        terminated = self.t >= c.horizon
        truncated = False
        if terminated:
            reward += self._liquidate()       # pay to flatten any residual inventory
        else:
            self._cancel_agent_quotes()
            self._post_background(self._fair)  # fresh ladder around new fair value

        return obs, float(reward), terminated, truncated, self._info()

    # ------------------------------------------------------------------ #
    #  Internals
    # ------------------------------------------------------------------ #

    def _reset_runtime(self) -> None:
        self.t = 0
        self.cash = 0.0
        self.inventory = 0
        self._fair = self.cfg.s0
        self._bg_orders: list[int] = []
        self._mm_orders: dict[int, Side] = {}
        self._mid_returns: deque[float] = deque(maxlen=self.cfg.vol_window)
        self._step_fills = 0
        self._prev_mid = self.cfg.s0
        self._prev_wealth = 0.0
        self._replay: np.ndarray | None = None  # per-episode replay window, if any

    # --- fill accounting via the engine's trade callback ---
    def _on_trade(self, trade: Trade) -> None:
        side = self._mm_orders.get(trade.maker_id)
        if side is None:
            return  # a background-vs-background trade; irrelevant to the agent
        if side is Side.BID:                 # our buy quote was hit -> we bought
            self.inventory += trade.quantity
            self.cash -= trade.quantity * trade.price
        else:                                # our sell quote was hit -> we sold
            self.inventory -= trade.quantity
            self.cash += trade.quantity * trade.price
        self._step_fills += trade.quantity

    # --- background liquidity ---
    def _post_background(self, mid: float) -> None:
        c = self.cfg
        self._cancel_background()
        mid = round(mid / c.tick) * c.tick
        for i in range(c.bg_min_offset, c.bg_min_offset + c.bg_depth):
            bid_p = round(mid - i * c.tick, c.price_decimals)
            ask_p = round(mid + i * c.tick, c.price_decimals)
            ob, _ = self.engine.submit(Side.BID, c.bg_level_size, price=bid_p)
            oa, _ = self.engine.submit(Side.ASK, c.bg_level_size, price=ask_p)
            self._bg_orders.extend((ob.id, oa.id))

    def _cancel_background(self) -> None:
        for oid in self._bg_orders:
            self.engine.cancel(oid)
        self._bg_orders.clear()

    # --- agent quoting ---
    def _decode_action(self, action) -> tuple[int, int]:
        """Map a raw action to integer (bid, ask) tick offsets for both modes."""
        c = self.cfg
        if c.continuous_action:
            a = np.asarray(action, dtype=np.float32).reshape(-1)
            # [-1, 1] -> [1, n_offsets] ticks, then snap to the tick grid.
            span = c.n_offsets - 1
            bid_t = 1 + (float(np.clip(a[0], -1, 1)) + 1) / 2 * span
            ask_t = 1 + (float(np.clip(a[1], -1, 1)) + 1) / 2 * span
            return int(round(bid_t)), int(round(ask_t))
        bid_idx, ask_idx = divmod(int(action), c.n_offsets)
        return bid_idx + 1, ask_idx + 1

    def _place_agent_quotes(self, ref_mid: float, bid_ticks: int, ask_ticks: int) -> None:
        c = self.cfg
        bid_off = bid_ticks * c.tick
        ask_off = ask_ticks * c.tick
        self._mm_orders = {}

        # Hard risk limit: size each quote down so a fill can never breach the cap.
        buy_room = c.max_inventory - self.inventory       # shares we may still buy
        sell_room = c.max_inventory + self.inventory      # shares we may still sell
        if buy_room > 0:
            bid_p = round(ref_mid - bid_off, c.price_decimals)
            o, _ = self.engine.submit(Side.BID, min(c.mm_order_size, buy_room), price=bid_p)
            if not o.is_filled:
                self._mm_orders[o.id] = Side.BID
        if sell_room > 0:
            ask_p = round(ref_mid + ask_off, c.price_decimals)
            o, _ = self.engine.submit(Side.ASK, min(c.mm_order_size, sell_room), price=ask_p)
            if not o.is_filled:
                self._mm_orders[o.id] = Side.ASK

    def _cancel_agent_quotes(self) -> None:
        for oid in self._mm_orders:
            self.engine.cancel(oid)
        self._mm_orders = {}

    # --- background market-order flow ---
    def _run_market_orders(self, planned_increment: float) -> int:
        c = self.cfg
        n = self._sample_arrivals()
        bias = 0
        if planned_increment > 0:
            bias = 1
        elif planned_increment < 0:
            bias = -1

        net_signed = 0
        for _ in range(n):
            informed = self.np_random.random() < c.informed_ratio and bias != 0
            buy = (bias > 0) if informed else (self.np_random.random() < 0.5)
            size = int(self.np_random.integers(1, c.max_market_order_size + 1))
            side = Side.BID if buy else Side.ASK   # market BUY lifts asks, etc.
            self.engine.submit(side, size, order_type=OrderType.MARKET)
            net_signed += size if buy else -size
        return net_signed

    def _sample_arrivals(self) -> int:
        c = self.cfg
        if not c.use_hawkes:
            return int(self.np_random.poisson(c.arrival_rate))
        # discrete-time Hawkes: decay toward mu, sample, then self-excite.
        self._hawkes_intensity = c.hawkes_mu + (
            self._hawkes_intensity - c.hawkes_mu
        ) * math.exp(-c.hawkes_beta)
        n = int(self.np_random.poisson(self._hawkes_intensity))
        self._hawkes_intensity += c.hawkes_alpha * n
        return n

    # --- terminal liquidation ---
    def _liquidate(self) -> float:
        """Flatten any residual inventory at the horizon; return the reward delta.

        We model the unwind analytically - cross the spread at a fixed cost of
        ``bg_min_offset`` ticks per share around fair value - rather than firing
        a market order at whatever depth happens to remain. This guarantees a
        flat book and charges a realistic liquidity-taking cost, so the agent is
        taught to run inventory down *before* the horizon instead of dumping it.
        """
        before = self._prev_wealth
        q = self.inventory
        if q != 0:
            cost = self.cfg.bg_min_offset * self.cfg.tick   # spread paid per share
            exec_price = self._fair - math.copysign(cost, q)  # sell low / buy high
            self.cash += q * exec_price
            self.inventory = 0
        wealth = self._wealth()
        # reward delta from liquidation: ~ -|q| * cost (negative -> pay to flatten).
        return (wealth - before) - self.cfg.inventory_penalty * self.inventory**2

    # --- observation / bookkeeping helpers ---
    def _observed_mid(self) -> float:
        mid = self.engine.mid()
        return mid if mid is not None else self._fair

    def _wealth(self) -> float:
        return self.cash + self.inventory * self._fair

    def _build_obs(self, reward_return: float) -> np.ndarray:
        c = self.cfg
        spread = self.engine.spread()
        spread_ticks = (spread / c.tick) if spread is not None else 2 * c.bg_min_offset
        vol = float(np.std(self._mid_returns)) if len(self._mid_returns) > 1 else c.sigma

        obs = np.array(
            [
                self.inventory / c.max_inventory,            # inventory (signed, ~[-1,1])
                self.engine.book.imbalance(c.bg_depth),      # book imbalance [-1,1]
                spread_ticks / (2 * (c.bg_min_offset + c.bg_depth)),  # spread, normalised
                (c.horizon - self.t) / c.horizon,            # time remaining [0,1]
                np.clip(reward_return / (c.sigma + 1e-9), -10, 10),   # recent return (z)
                np.clip(vol / (c.sigma + 1e-9), 0, 10),      # realised vol (relative)
            ],
            dtype=np.float32,
        )
        return obs

    def _info(self) -> dict:
        return {
            "t": self.t,
            "inventory": self.inventory,
            "cash": self.cash,
            "wealth": self._wealth(),
            "fair": self._fair,
            "mid": self._observed_mid(),
            "fills": self._step_fills,
        }
