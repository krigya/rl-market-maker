"""Environment-contract tests.

These guard the properties the RL training will silently depend on: a valid
Gymnasium interface, reproducibility, the hard inventory risk limit, terminal
flattening, and a reward that actually punishes inventory.
"""

import numpy as np

from quant_mm.agents import AvellanedaStoikovAgent, FixedSymmetricAgent, RandomAgent
from quant_mm.sim import MMEnvConfig, MarketMakingEnv


def small_cfg(**kw) -> MMEnvConfig:
    return MMEnvConfig(horizon=50, **kw)


def test_reset_and_step_obey_gym_api() -> None:
    env = MarketMakingEnv(small_cfg())
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert isinstance(info, dict)

    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert env.observation_space.contains(obs)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)


def test_episode_runs_for_horizon_steps() -> None:
    cfg = small_cfg()
    env = MarketMakingEnv(cfg)
    env.reset(seed=1)
    steps = 0
    done = False
    while not done:
        _, _, term, trunc, _ = env.step(0)
        steps += 1
        done = term or trunc
    assert steps == cfg.horizon


def test_determinism_same_seed_same_trajectory() -> None:
    cfg = small_cfg()
    env = MarketMakingEnv(cfg)

    def rollout(seed):
        obs, _ = env.reset(seed=seed)
        rewards = []
        done = False
        while not done:
            obs, r, term, trunc, _ = env.step(5)  # fixed action
            rewards.append(r)
            done = term or trunc
        return rewards

    assert rollout(123) == rollout(123)


def test_hard_inventory_limit_is_never_breached() -> None:
    cfg = small_cfg(max_inventory=20, informed_ratio=0.9, arrival_rate=3.0)
    env = MarketMakingEnv(cfg)
    agent = RandomAgent(cfg, seed=7)
    for ep in range(10):
        obs, _ = env.reset(seed=ep)
        agent.reset()
        done = False
        while not done:
            obs, _, term, trunc, info = env.step(agent.act(obs))
            assert abs(info["inventory"]) <= cfg.max_inventory
            done = term or trunc


def test_terminal_liquidation_flattens_inventory() -> None:
    cfg = small_cfg()
    env = MarketMakingEnv(cfg)
    # An always-quote agent will accumulate inventory; it must be flat at the end.
    obs, _ = env.reset(seed=2)
    done = False
    info = {}
    while not done:
        obs, _, term, trunc, info = env.step(0)  # tight quotes -> lots of trading
        done = term or trunc
    assert info["inventory"] == 0


def test_inventory_penalty_reduces_reward() -> None:
    """With a large penalty coefficient, carrying inventory must hurt the reward."""
    base = MarketMakingEnv(small_cfg(inventory_penalty=0.0))
    pen = MarketMakingEnv(small_cfg(inventory_penalty=1.0))

    def total_reward(env):
        obs, _ = env.reset(seed=3)
        tot = 0.0
        done = False
        while not done:
            obs, r, term, trunc, _ = env.step(0)  # tight quotes -> builds inventory
            tot += r
            done = term or trunc
        return tot

    assert total_reward(pen) < total_reward(base)


def test_as_agent_controls_inventory_better_than_random() -> None:
    """Sanity: the inventory-aware baseline should hold less inventory than random."""
    from quant_mm.eval import evaluate

    cfg = small_cfg()
    env = MarketMakingEnv(cfg)
    as_inv = evaluate(env, AvellanedaStoikovAgent(cfg), n_episodes=30)["mean_abs_inventory"]
    rnd_inv = evaluate(env, RandomAgent(cfg, seed=0), n_episodes=30)["mean_abs_inventory"]
    assert as_inv < rnd_inv


def test_observation_is_finite_and_typed() -> None:
    env = MarketMakingEnv(small_cfg())
    obs, _ = env.reset(seed=0)
    agent = FixedSymmetricAgent(env.cfg, 2)
    done = False
    while not done:
        obs, _, term, trunc, _ = env.step(agent.act(obs))
        assert obs.dtype == np.float32
        assert np.all(np.isfinite(obs))
        done = term or trunc
