"""Evaluation harness: roll an agent through the env and score it.

Any object with an ``act(obs) -> int`` method (the AS baseline, DQN, PPO, a
fixed policy) can be evaluated the same way, so comparisons are apples-to-apples.
The metrics are the ones a trading desk actually cares about:

* **terminal PnL** and its distribution across episodes (skill *and* consistency),
* **Sharpe** = mean / std of terminal PnL (risk-adjusted return - the number
  that decides whether a strategy is worth running),
* **inventory risk** taken: mean and max absolute inventory during the episode,
* **fill rate**: how much the agent actually traded.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Episode:
    """Per-step trajectory of a single episode (for plotting and diagnostics)."""

    inventory: list[float] = field(default_factory=list)
    wealth: list[float] = field(default_factory=list)
    mid: list[float] = field(default_factory=list)
    fair: list[float] = field(default_factory=list)
    reward: list[float] = field(default_factory=list)
    action: list[int] = field(default_factory=list)
    fills: list[int] = field(default_factory=list)
    terminal_pnl: float = 0.0


def run_episode(env, agent, seed: int | None = None) -> Episode:
    """Run one episode of ``agent`` in ``env`` and record the trajectory."""
    obs, info = env.reset(seed=seed)
    if hasattr(agent, "reset"):
        agent.reset()
    ep = Episode()
    done = False
    while not done:
        action = agent.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        ep.inventory.append(info["inventory"])
        ep.wealth.append(info["wealth"])
        ep.mid.append(info["mid"])
        ep.fair.append(info["fair"])
        ep.reward.append(reward)
        # discrete actions are ints; continuous actions are small arrays.
        ep.action.append(int(action) if np.ndim(action) == 0 else np.asarray(action).tolist())
        ep.fills.append(info["fills"])
        done = terminated or truncated
    ep.terminal_pnl = ep.wealth[-1] if ep.wealth else 0.0
    return ep


def evaluate(env, agent, n_episodes: int = 200, seed0: int = 10_000) -> dict:
    """Aggregate metrics over ``n_episodes`` (deterministic seeds for reproducibility)."""
    pnls, mean_abs_inv, max_abs_inv, total_fills = [], [], [], []
    for i in range(n_episodes):
        ep = run_episode(env, agent, seed=seed0 + i)
        pnls.append(ep.terminal_pnl)
        inv = np.abs(ep.inventory)
        mean_abs_inv.append(inv.mean() if len(inv) else 0.0)
        max_abs_inv.append(inv.max() if len(inv) else 0.0)
        total_fills.append(int(np.sum(ep.fills)))

    pnls = np.asarray(pnls)
    std = pnls.std()
    return {
        "n_episodes": n_episodes,
        "pnl_mean": float(pnls.mean()),
        "pnl_std": float(std),
        "pnl_median": float(np.median(pnls)),
        "sharpe": float(pnls.mean() / std) if std > 1e-12 else float("nan"),
        "pnl_p05": float(np.percentile(pnls, 5)),
        "pnl_p95": float(np.percentile(pnls, 95)),
        "win_rate": float((pnls > 0).mean()),
        "mean_abs_inventory": float(np.mean(mean_abs_inv)),
        "max_abs_inventory": float(np.mean(max_abs_inv)),
        "avg_fills": float(np.mean(total_fills)),
        "pnls": pnls,
    }


def summary_table(results: dict[str, dict]) -> str:
    """Render a dict of ``{agent_name: evaluate(...)}`` as a readable table."""
    cols = ["pnl_mean", "pnl_std", "sharpe", "win_rate", "mean_abs_inventory", "avg_fills"]
    header = f"{'agent':<22}" + "".join(f"{c:>20}" for c in cols)
    lines = [header, "-" * len(header)]
    for name, r in results.items():
        row = f"{name:<22}" + "".join(f"{r[c]:>20.4f}" for c in cols)
        lines.append(row)
    return "\n".join(lines)
