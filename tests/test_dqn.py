"""DQN component tests.

Fast, deterministic checks on the moving parts of the agent -- network shapes,
replay buffer mechanics, a gradient step, the epsilon schedule, greedy-action
validity, and save/load -- without any expensive training.
"""

import numpy as np
import torch

from quant_mm.agents import DQN, DQNConfig, QNetwork, ReplayBuffer


def test_qnetwork_output_shape() -> None:
    net = QNetwork(obs_dim=6, n_actions=16)
    out = net(torch.zeros(4, 6))
    assert out.shape == (4, 16)


def test_replay_buffer_push_and_sample() -> None:
    buf = ReplayBuffer(capacity=100, obs_dim=3, seed=0)
    for i in range(150):  # overflow capacity to exercise the ring
        buf.push(np.ones(3) * i, i % 4, float(i), np.ones(3) * (i + 1), i % 2)
    assert len(buf) == 100
    s, a, r, s2, d = buf.sample(16)
    assert s.shape == (16, 3) and a.shape == (16,)
    assert s2.shape == (16, 3) and r.shape == (16,) and d.shape == (16,)
    assert a.dtype == torch.int64


def test_epsilon_decays_monotonically() -> None:
    agent = DQN(6, 16, DQNConfig(eps_start=1.0, eps_end=0.05, eps_decay_steps=1000))
    agent.total_steps = 0
    e0 = agent.epsilon()
    agent.total_steps = 500
    e1 = agent.epsilon()
    agent.total_steps = 5000
    e2 = agent.epsilon()
    assert e0 > e1 > e2
    assert abs(e0 - 1.0) < 1e-9 and abs(e2 - 0.05) < 1e-9  # clamps at the ends


def test_greedy_action_is_valid() -> None:
    agent = DQN(6, 16, DQNConfig())
    a = agent.act(np.zeros(6, dtype=np.float32))
    assert isinstance(a, int) and 0 <= a < 16


def test_learn_step_runs_and_changes_weights() -> None:
    cfg = DQNConfig(learning_starts=8, batch_size=8, seed=0)
    agent = DQN(6, 16, cfg)
    rng = np.random.default_rng(0)
    for _ in range(64):
        agent.buffer.push(
            rng.standard_normal(6), rng.integers(16), rng.standard_normal(),
            rng.standard_normal(6), rng.integers(2),
        )
    before = agent.online.net[0].weight.detach().clone()
    loss = agent.learn_step()
    after = agent.online.net[0].weight.detach()
    assert loss is not None and np.isfinite(loss)
    assert not torch.allclose(before, after)  # a gradient step actually moved weights


def test_target_sync_copies_weights() -> None:
    agent = DQN(6, 16, DQNConfig(target_update_freq=1))
    # perturb the online net, then sync
    with torch.no_grad():
        agent.online.net[0].weight.add_(1.0)
    agent.total_steps = 1
    agent.maybe_sync_target()
    for po, pt in zip(agent.online.parameters(), agent.target.parameters()):
        assert torch.allclose(po, pt)


def test_save_and_load_roundtrip(tmp_path) -> None:
    agent = DQN(6, 16, DQNConfig(seed=1))
    obs = np.random.default_rng(0).standard_normal(6).astype(np.float32)
    a_before = agent.act(obs)
    path = str(tmp_path / "dqn.pt")
    agent.save(path)
    restored = DQN.load(path)
    assert restored.act(obs) == a_before
