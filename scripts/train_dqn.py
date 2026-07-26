"""Train the DQN market maker and save the agent + learning history.

Usage (from the project root, using the project venv)::

    .venv\\Scripts\\python.exe scripts\\train_dqn.py --episodes 1500

Artifacts are written to ``runs/dqn/``:
    * ``dqn.pt``       -- the trained Q-network (load with ``DQN.load``)
    * ``history.json`` -- per-episode training / evaluation metrics for plotting
"""

from __future__ import annotations

import argparse
import json
import pathlib

import torch

from quant_mm.agents import DQNConfig, train_dqn
from quant_mm.sim import MMEnvConfig, MarketMakingEnv


def main() -> None:
    p = argparse.ArgumentParser(description="Train the DQN market maker")
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="runs/dqn")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--inventory-penalty", type=float, default=None,
                   help="override the env inventory penalty (default: MMEnvConfig default)")
    args = p.parse_args()

    # The networks are tiny; multi-threaded BLAS only adds overhead here. Pinning
    # to one thread is ~10x faster for this workload on CPU.
    torch.set_num_threads(1)

    env_kw = {}
    if args.inventory_penalty is not None:
        env_kw["inventory_penalty"] = args.inventory_penalty
    env = MarketMakingEnv(MMEnvConfig(**env_kw))
    cfg = DQNConfig(seed=args.seed, lr=args.lr)

    # train_dqn checkpoints the best-by-Sharpe model and history into `out` as it
    # goes, so an interrupted run still leaves the best policy on disk.
    _, history = train_dqn(
        env, cfg, n_episodes=args.episodes, eval_every=args.eval_every, save_dir=args.out
    )

    out = pathlib.Path(args.out)
    (out / "history.json").write_text(json.dumps(history))
    print(f"\nBest agent + history saved to {out.resolve()}")


if __name__ == "__main__":
    main()
