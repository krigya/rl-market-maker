# A Limit-Order-Book Matching Engine and a Reinforcement-Learning Market Maker

**A from-scratch electronic market and three market-making agents — an analytic
optimum and two reinforcement learners — compared on risk-adjusted PnL.**

---

## 1. Overview

Market making is the business of continuously quoting a price to buy (bid) and a
price to sell (ask), earning the **spread** from trade flow rather than betting
on direction. The difficulty is entirely in the risk: ending up holding too much
of the asset (**inventory risk**), trading against better-informed
counterparties (**adverse selection**), and the fact that your own quotes move
the market.

This project builds the whole stack from first principles:

1. a **matching engine** with price-time priority, partial fills, and cancels;
2. a **stochastic market simulator** in which fills, market impact, and adverse
   selection *emerge* from real order interaction rather than being assumed;
3. the **Avellaneda–Stoikov** closed-form optimal quoter as an analytic baseline;
4. two reinforcement learners built from scratch — a **DQN** (value-based,
   discrete actions) and **PPO** (policy-gradient, continuous actions);
5. an **out-of-distribution validation** on a real (AAPL) price path.

Everything is tested (44 unit tests) and reproducible; each phase has a notebook
that runs end-to-end.

---

## 2. The matching engine

A limit order book must do four things quickly: read the best bid/ask, insert a
resting order, cancel an arbitrary order, and match the front order at the best
price. The design achieves all four cheaply:

* **Intrusive doubly-linked price levels.** Each price is a FIFO queue whose
  nodes *are* the orders. Append and unlink are O(1), so **cancellation is O(1)**
  (unlink a known node) rather than the O(n) scan a list would need. FIFO order
  is exactly time priority — the order resting longest fills first.
* **Lazy-deletion price heaps.** The best price is a heap root: O(1) to read,
  O(log n) to add a new level. Emptied levels are dropped from the dict and their
  stale prices discarded when they surface at the root, so cancels never touch
  the heap.
* **Sequence numbers, not wall-clock time**, encode time priority — giving fully
  deterministic, reproducible runs (essential once RL depends on the market).

Correctness is pinned by tests asserting the defining invariants: better prices
fill first, equal prices fill in arrival order, partial fills leave the correct
residual, market orders sweep levels and never rest, cancelled orders are
skipped. *(`src/quant_mm/lob/`, `tests/test_book.py`, `tests/test_engine.py`.)*

---

## 3. The market simulator

Most RL market-making demos assume a fill probability
`λ(δ) = A·e^{−κδ}`. This project does not — it lets that intensity emerge:

* The **fundamental** (fair value) follows an arithmetic Brownian motion
  `dS = σ dW`, the same assumption as Avellaneda–Stoikov, plus a linear
  **permanent market impact** term so net order flow nudges fair value.
* **Background liquidity** posts a ladder of limit orders each step; the agent's
  quotes rest in the *same* book and compete for queue position.
* **Liquidity-taking market orders** arrive as a **Poisson** (or self-exciting
  **Hawkes**) process and walk the book by price-time priority.

Two consequences fall out for free:

* **The fill/margin tradeoff is emergent.** Quoting tighter jumps ahead of
  background liquidity — more fills, less edge; quoting wider earns more per fill
  but rarely trades. Measured fill rate decays roughly exponentially with quoted
  distance, recovering the AS intensity `λ(δ)=A·e^{−κδ}`, with a genuine interior
  optimum (~3 ticks in the default calibration).
* **Adverse selection is observed, not coded.** A fraction of market orders are
  *informed* (direction correlated with the next fundamental move), so the agent
  that gets filled is disproportionately filled right before the price moves
  against it. Raising the informed fraction erodes naive tight quoting faster
  than the inventory-aware baseline.

### RL formulation

* **State** `[inventory, book imbalance, spread, time-remaining, recent return,
  realised volatility]`, all normalised.
* **Action** the quote offsets — a discrete `(bid, ask)` tick grid for DQN, a
  continuous 2-D vector for PPO. Asymmetric offsets let the agent **skew**
  against inventory.
* **Reward** `r_t = ΔWealth_t − λ·q_t²` each step (mark-to-fair-value PnL minus a
  quadratic inventory penalty), with a forced **terminal liquidation** so
  unflattened inventory is paid for. A **hard risk limit** sizes quotes down near
  the position cap, as a real desk would.

*(`src/quant_mm/sim/`, `tests/test_env.py`.)*

---

## 4. The Avellaneda–Stoikov baseline

The analytic optimum from stochastic control. Around mid `s`, with inventory `q`
and time-to-horizon `(T−t)`, quote around the **reservation price**

$$ r(s,t) = s - q\,\gamma\,\sigma^2\,(T-t) $$

with optimal total spread

$$ \delta = \gamma\sigma^2(T-t) + \tfrac{2}{\gamma}\ln\!\big(1+\tfrac{\gamma}{\kappa}\big), $$

quoting `bid = r − δ/2`, `ask = r + δ/2`. When long (`q>0`) the reservation price
sits below mid, so both quotes shift down — keener to sell, shyer to buy — which
bleeds the position off. That skew *is* inventory management. This is the bar the
RL agents must clear. *(`src/quant_mm/agents/baseline.py`.)*

---

## 5. The reinforcement learners

**DQN (value-based, discrete).** Learns `Q(s,a)` and acts greedily. Built from
scratch with the three ingredients that make it stable — **experience replay**
(decorrelate consecutive market steps), a **target network** (stop chasing a
moving bootstrap target), and **ε-greedy** exploration — plus **Double DQN**
(curbs Q-overestimation) and a **Huber** loss. Trained 1000 episodes; Sharpe
peaked mid-training then drifted (classic DQN instability), which is exactly why
the **best out-of-sample checkpoint** is kept.

**PPO (policy-gradient, continuous).** An **actor-critic** outputs a Gaussian
over the continuous `(bid, ask)` offsets; improved with **Generalized Advantage
Estimation** and the **clipped surrogate objective** (a trust region that permits
several optimisation epochs per batch), with an entropy bonus. The continuous
action is the natural fit for "how far to quote." *(`src/quant_mm/agents/dqn.py`,
`ppo.py`; trainers in `scripts/`.)*

---

## 6. Results

### In-distribution (synthetic market, 300 held-out episodes, identical seeds)

| Agent | PnL | Sharpe | mean \|inventory\| | win rate |
|---|---:|---:|---:|---:|
| Fixed (3 ticks) | 11.58 | 2.75 | 19.7 | 99% |
| **Avellaneda–Stoikov** | 10.39 | **6.49** | 6.63 | 100% |
| **PPO** (continuous) | 11.09 | **6.34** | 7.58 | 100% |
| **DQN** (discrete) | 9.81 | 6.06 | 5.90 | 100% |

Reading: a naive fixed quoter makes plenty of PnL but at 3× the inventory and a
quarter of the risk-adjusted return. Both RL agents, trained on reward alone and
never shown the AS formula, land in the neighbourhood of the analytic optimum.
**PPO's continuous policy captures more PnL than DQN** (11.1 vs 9.8); DQN holds
**tighter inventory** (5.9 vs 7.6). Neither beats AS on Sharpe here — an honest
result, and the open thread for further tuning.

### Out-of-distribution (AAPL daily path replayed as the fundamental, 200 episodes)

The real series is markedly non-Gaussian (excess kurtosis ≈ 2.7; positive
autocorrelation of |returns| — volatility clustering). Yet:

| Agent | real-path Sharpe |
|---|---:|
| Avellaneda–Stoikov | 6.79 |
| DQN | 6.63 |
| PPO | 6.34 |

Risk-adjusted performance holds — DQN's even rises — because the policies condition
on *inventory and microstructure state*, not on the specific return law.

---

## 7. Limitations and extensions

* **Partial sim-to-real.** Phase 5 replays the real fundamental path but keeps
  synthetic flow and standardises volatility. A full study would replay a real
  limit-order book (LOBSTER) end-to-end; the loader is already CSV-compatible.
* **Coarse action grid.** Offsets span 1–4 ticks; a finer grid or a larger
  continuous range would sharpen the DQN-vs-PPO contrast.
* **DQN stability.** Best-checkpointing masks but does not fix the mid-training
  drift; Polyak target updates, prioritized replay, or a dueling head are natural
  next steps — and the explicit route to *beating the original DQN*.
* **Publishable directions.** An inventory-aware adverse-selection penalty; a
  non-stationary (Hawkes-regime-switching) stress test; multi-asset/hedging
  extensions — all consistent with live HFT research.

---

## 8. Interview talking points

* Why price-time priority, and which data structures give O(1) cancel / O(1)
  best-price / O(log n) insert.
* Deriving the AS reservation price and *why* it skews quotes against inventory.
* DQN: what replay and the target network each fix, and why Double DQN reduces
  overestimation.
* PPO: why policy-gradient suits continuous quoting, and what the clip does.
* Reward shaping: why the inventory penalty and terminal liquidation exist, and
  the fight to scale them.
* Non-stationarity, the self-impact problem, and the sim-to-real gap made
  concrete in Phase 5.

## References

- Avellaneda, M. & Stoikov, S. (2008). *High-frequency trading in a limit order
  book.* Quantitative Finance 8(3).
- Mnih et al. (2015). *Human-level control through deep reinforcement learning*
  (DQN). Nature.
- van Hasselt, Guez & Silver (2016). *Deep RL with Double Q-learning.*
- Schulman et al. (2017). *Proximal Policy Optimization Algorithms.*
- Schulman et al. (2016). *High-Dimensional Continuous Control Using GAE.*
