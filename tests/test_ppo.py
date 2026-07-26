"""PPO component tests: network, GAE, action interface, continuous env decode."""

import numpy as np
import torch

from quant_mm.agents import PPO, ActorCritic, PPOConfig
from quant_mm.agents.ppo import _compute_gae
from quant_mm.sim import MMEnvConfig, MarketMakingEnv


def test_actorcritic_shapes() -> None:
    net = ActorCritic(obs_dim=6, act_dim=2)
    mean, value = net(torch.zeros(8, 6))
    assert mean.shape == (8, 2)
    assert value.shape == (8,)
    dist = net.distribution(torch.zeros(8, 6))
    assert dist.sample().shape == (8, 2)


def test_act_returns_clipped_action() -> None:
    agent = PPO(6, 2, PPOConfig())
    a = agent.act(np.zeros(6, dtype=np.float32))
    assert a.shape == (2,)
    assert np.all(a >= -1.0) and np.all(a <= 1.0)


def test_compute_gae_known_values() -> None:
    rewards = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    values = np.zeros(3, dtype=np.float32)
    dones = np.zeros(3, dtype=np.float32)
    adv, returns = _compute_gae(rewards, values, dones, last_value=0.0, gamma=1.0, lam=1.0)
    # With zero baselines and no discounting, advantage = remaining reward-to-go.
    assert np.allclose(adv, [3.0, 2.0, 1.0])
    assert np.allclose(returns, [3.0, 2.0, 1.0])


def test_gae_respects_episode_boundaries() -> None:
    rewards = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    values = np.zeros(3, dtype=np.float32)
    dones = np.array([0.0, 1.0, 0.0], dtype=np.float32)  # episode ends AT t=1
    adv, _ = _compute_gae(rewards, values, dones, last_value=0.0, gamma=1.0, lam=1.0)
    # t=0 and t=1 are the same episode -> adv[0] = r0 + r1 = 2. The terminal at
    # t=1 stops bootstrapping into t=2, so adv[1] = r1 = 1 (does NOT include r2).
    assert np.isclose(adv[0], 2.0)
    assert np.isclose(adv[1], 1.0)


def test_continuous_env_action_decode() -> None:
    cfg = MMEnvConfig(continuous_action=True)
    env = MarketMakingEnv(cfg)
    assert env.action_space.shape == (2,)
    # extremes map to the tightest / widest tick offsets
    assert env._decode_action(np.array([-1.0, -1.0])) == (1, 1)
    assert env._decode_action(np.array([1.0, 1.0])) == (cfg.n_offsets, cfg.n_offsets)


def test_save_load_roundtrip(tmp_path) -> None:
    agent = PPO(6, 2, PPOConfig(seed=3))
    obs = np.random.default_rng(0).standard_normal(6).astype(np.float32)
    before = agent.act(obs)
    path = str(tmp_path / "ppo.pt")
    agent.save(path)
    restored = PPO.load(path)
    assert np.allclose(restored.act(obs), before)


def test_ppo_runs_in_continuous_env() -> None:
    env = MarketMakingEnv(MMEnvConfig(continuous_action=True, horizon=30))
    agent = PPO(env.observation_space.shape[0], env.action_space.shape[0])
    obs, _ = env.reset(seed=0)
    done = False
    while not done:
        obs, r, term, trunc, _ = env.step(agent.act(obs))
        assert np.isfinite(r)
        done = term or trunc
