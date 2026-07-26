"""Train the PPO market maker (continuous quote offsets) and save artifacts.

Usage (from the project root, using the project venv)::

    .venv\\Scripts\\python.exe scripts\\train_ppo.py --iters 120

Writes ``runs/ppo/{ppo.pt, history.json}`` (best-by-Sharpe checkpoint).
"""

from __future__ import annotations

import argparse
import json
import pathlib

import torch

from quant_mm.agents import PPOConfig, train_ppo
from quant_mm.sim import MMEnvConfig, MarketMakingEnv


def main() -> None:
    p = argparse.ArgumentParser(description="Train the PPO market maker")
    p.add_argument("--iters", type=int, default=120)
    p.add_argument("--steps-per-iter", type=int, default=4096)
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="runs/ppo")
    args = p.parse_args()

    torch.set_num_threads(1)  # tiny nets -> single thread is fastest on CPU

    env = MarketMakingEnv(MMEnvConfig(continuous_action=True))
    cfg = PPOConfig(seed=args.seed)

    _, history = train_ppo(
        env, cfg,
        n_iterations=args.iters,
        steps_per_iter=args.steps_per_iter,
        eval_every=args.eval_every,
        save_dir=args.out,
    )

    out = pathlib.Path(args.out)
    (out / "history.json").write_text(json.dumps(history))
    print(f"\nBest PPO agent + history saved to {out.resolve()}")


if __name__ == "__main__":
    main()
