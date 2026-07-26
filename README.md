# quant-mm - Limit-Order-Book Matching Engine + RL Market Maker

A from-scratch electronic market and a reinforcement-learning market-making agent.

## The idea in one paragraph

A **market maker** continuously quotes a price to buy (bid) and a price to sell (ask),
earning the **spread** between them off the flow of trades - not off betting on direction.
The hard part is risk management: ending up holding too much of the asset (**inventory
risk**), trading against better-informed counterparties (**adverse selection**), and the
fact that your own quotes move the market. This project builds (1) the exchange machinery
that matches orders by **price-time priority**, (2) a simulated market to trade in, (3) the
**Avellaneda-Stoikov** closed-form quoting rule as an analytic baseline, and (4) RL agents
(DQN, then PPO) that learn to quote and manage inventory - and should beat that baseline.

## Layout

| Path | Contents |
|------|----------|
| `src/quant_mm/lob/`   | Matching engine: order types, the order book, the engine API |
| `src/quant_mm/sim/`   | Synthetic stochastic market + Gymnasium env + historical replay |
| `src/quant_mm/agents/`| Avellaneda-Stoikov baseline, DQN, PPO |
| `src/quant_mm/eval/`  | Metrics (PnL, Sharpe, inventory) and plots |
| `tests/`              | Engine-invariant and environment-contract tests |
| `notebooks/`          | Phase-by-phase demos and results |
| `report/`             | Writeup: math, design, results, limitations |

## Setup (Windows, project-local venv)

```powershell
py -m venv .venv
.venv\Scripts\python.exe -m pip install -e .[dev]
.venv\Scripts\python.exe -m pytest        # 44 tests
```

## The model (reference)

- **State**: `[inventory, book imbalance, spread, time-to-horizon, recent return, volatility]`
- **Action**: quote offsets - discrete tick grid (DQN) or continuous 2-D (PPO)
- **Reward**: `r_t = ΔWealth_t - λ·q²` plus a terminal liquidation penalty
- **Avellaneda-Stoikov reservation price**: `r(s, t) = s - q·γ·σ²·(T - t)`

## Reproduce

```powershell
# train the agents (artifacts land in runs/)
.venv\Scripts\python.exe scripts\train_dqn.py --episodes 1000
.venv\Scripts\python.exe scripts\train_ppo.py --iters 120
# then run the notebooks 01..05 in order (each runs top-to-bottom)
```

## Headline result

Risk-adjusted PnL (Sharpe) over 300 held-out episodes on identical market seeds:

| Agent | PnL | Sharpe | mean \|inventory\| |
|---|---:|---:|---:|
| Fixed (3 ticks) | 11.58 | 2.75 | 19.7 |
| Avellaneda-Stoikov | 10.39 | **6.49** | 6.63 |
| PPO (continuous) | 11.09 | 6.34 | 7.58 |
| DQN (discrete) | 9.81 | 6.06 | 5.90 |

Both from-scratch RL agents reach the neighbourhood of the analytic optimum, and
hold their performance on a real (fat-tailed) AAPL price path (Phase 5).

## Status

All phases complete. Full write-up in [`report/report.md`](report/report.md);
phase-by-phase notebooks in [`notebooks/`](notebooks/).
