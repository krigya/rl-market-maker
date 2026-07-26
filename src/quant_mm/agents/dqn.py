"""Deep Q-Network market maker, implemented from scratch in PyTorch.

DQN learns an **action-value function** ``Q(s, a)`` -- the expected discounted
future reward of taking action ``a`` in state ``s`` and acting greedily after.
The policy is then 'pick the action with the highest Q'. Three ingredients make
it train stably; each is here and each is worth being able to defend:

1. **Experience replay** (:class:`ReplayBuffer`). Transitions are stored and
   sampled in random minibatches. This breaks the temporal correlation between
   consecutive market steps (which would otherwise bias the gradient) and lets
   each experience be reused many times -- crucial when env steps are expensive.

2. **Target network**. The bootstrap target
   ``y = r + gamma * max_a' Q(s', a')`` is computed with a *frozen* copy of the
   network that is only periodically synced. Without it you are regressing
   toward a target that moves every gradient step -- a feedback loop that
   routinely diverges.

3. **Off-policy exploration** via epsilon-greedy with a decaying schedule: act
   randomly with probability ``epsilon`` (high early, low late) so the agent
   discovers the fill/inventory tradeoff before committing to a policy.

We also support **Double DQN**: decouple the action *selection* (online net)
from its *evaluation* (target net) in the bootstrap, which curbs the
well-known overestimation bias of vanilla DQN. Toggle with ``double_dqn``.

The loss is the **Huber** (smooth-L1) loss on the temporal-difference error,
which is less sensitive to the occasional large reward than plain MSE.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  Q-network
# --------------------------------------------------------------------------- #
class QNetwork(nn.Module):
    """A small MLP mapping a state to one Q-value per discrete action."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: tuple[int, ...] = (128, 128)):
        super().__init__()
        layers: list[nn.Module] = []
        last = obs_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers.append(nn.Linear(last, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# --------------------------------------------------------------------------- #
#  Replay buffer
# --------------------------------------------------------------------------- #
class ReplayBuffer:
    """Fixed-capacity ring buffer of transitions, sampled uniformly at random."""

    def __init__(self, capacity: int, obs_dim: int, seed: int | None = None):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.size = 0
        self.ptr = 0
        self.rng = np.random.default_rng(seed)

    def push(self, s, a, r, s2, done) -> None:
        i = self.ptr
        self.obs[i] = s
        self.actions[i] = a
        self.rewards[i] = r
        self.next_obs[i] = s2
        self.dones[i] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = self.rng.integers(0, self.size, size=batch_size)
        return (
            torch.from_numpy(self.obs[idx]),
            torch.from_numpy(self.actions[idx]),
            torch.from_numpy(self.rewards[idx]),
            torch.from_numpy(self.next_obs[idx]),
            torch.from_numpy(self.dones[idx]),
        )

    def __len__(self) -> int:
        return self.size


# --------------------------------------------------------------------------- #
#  Trainer
# --------------------------------------------------------------------------- #
@dataclass
class DQNConfig:
    hidden: tuple[int, ...] = (128, 128)
    lr: float = 1e-3
    gamma: float = 0.99               # RL discount (distinct from the env's risk aversion)
    buffer_size: int = 100_000
    batch_size: int = 128
    learning_starts: int = 2_000      # steps of random play before learning
    train_freq: int = 1               # gradient steps per env step
    target_update_freq: int = 1_000   # hard target-net sync interval (steps)
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 40_000     # linear epsilon decay horizon
    reward_scale: float = 1.0
    double_dqn: bool = True
    grad_clip: float = 10.0
    seed: int = 0


class DQN:
    """Trains and holds the Q-network; ``act`` exposes the greedy policy."""

    def __init__(self, obs_dim: int, n_actions: int, cfg: DQNConfig | None = None):
        self.cfg = cfg or DQNConfig()
        self.obs_dim = obs_dim
        self.n_actions = n_actions

        torch.manual_seed(self.cfg.seed)
        self.online = QNetwork(obs_dim, n_actions, self.cfg.hidden)
        self.target = QNetwork(obs_dim, n_actions, self.cfg.hidden)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.opt = torch.optim.Adam(self.online.parameters(), lr=self.cfg.lr)
        self.buffer = ReplayBuffer(self.cfg.buffer_size, obs_dim, seed=self.cfg.seed)
        self.rng = np.random.default_rng(self.cfg.seed)
        self.total_steps = 0

    # --- policy ---
    def epsilon(self) -> float:
        c = self.cfg
        frac = min(1.0, self.total_steps / c.eps_decay_steps)
        return c.eps_start + frac * (c.eps_end - c.eps_start)

    def act(self, obs: np.ndarray) -> int:
        """Greedy action -- used at evaluation (no exploration)."""
        with torch.no_grad():
            q = self.online(torch.from_numpy(np.asarray(obs, dtype=np.float32)))
            return int(torch.argmax(q).item())

    def act_explore(self, obs: np.ndarray) -> int:
        """Epsilon-greedy action -- used while collecting training data."""
        if self.rng.random() < self.epsilon():
            return int(self.rng.integers(self.n_actions))
        return self.act(obs)

    def reset(self) -> None:  # uniform interface with the other agents
        pass

    # --- learning ---
    def learn_step(self) -> float | None:
        c = self.cfg
        if len(self.buffer) < max(c.batch_size, c.learning_starts):
            return None

        s, a, r, s2, done = self.buffer.sample(c.batch_size)
        r = r * c.reward_scale

        # Q(s, a) for the actions actually taken.
        q_sa = self.online(s).gather(1, a.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if c.double_dqn:
                # Online net picks the next action; target net evaluates it.
                next_a = self.online(s2).argmax(dim=1, keepdim=True)
                q_next = self.target(s2).gather(1, next_a).squeeze(1)
            else:
                q_next = self.target(s2).max(dim=1).values
            target = r + c.gamma * (1.0 - done) * q_next

        loss = F.smooth_l1_loss(q_sa, target)  # Huber on the TD error
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), c.grad_clip)
        self.opt.step()
        return float(loss.item())

    def maybe_sync_target(self) -> None:
        if self.total_steps % self.cfg.target_update_freq == 0:
            self.target.load_state_dict(self.online.state_dict())

    # --- persistence ---
    def save(self, path: str) -> None:
        torch.save(
            {
                "online": self.online.state_dict(),
                "cfg": self.cfg,
                "obs_dim": self.obs_dim,
                "n_actions": self.n_actions,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "DQN":
        ckpt = torch.load(path, weights_only=False)
        agent = cls(ckpt["obs_dim"], ckpt["n_actions"], ckpt["cfg"])
        agent.online.load_state_dict(ckpt["online"])
        agent.target.load_state_dict(ckpt["online"])
        return agent


# --------------------------------------------------------------------------- #
#  Training loop
# --------------------------------------------------------------------------- #
def train_dqn(
    env,
    cfg: DQNConfig | None = None,
    n_episodes: int = 1_500,
    eval_every: int = 100,
    eval_episodes: int = 60,
    verbose: bool = True,
    save_dir: str | None = None,
):
    """Train a DQN on ``env`` and return ``(agent, history)``.

    Training seeds and evaluation seeds are drawn from disjoint ranges so the
    reported eval performance is genuinely out-of-sample.

    If ``save_dir`` is given, the **best out-of-sample model** (by eval Sharpe) is
    checkpointed to ``save_dir/dqn.pt`` on each evaluation, and ``history.json``
    is written incrementally -- so an interrupted run still leaves usable
    artifacts, and we keep the best policy rather than the last one.
    """
    import json
    import pathlib

    from ..eval import evaluate  # local import avoids a circular dependency

    cfg = cfg or DQNConfig()
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent = DQN(obs_dim, n_actions, cfg)

    out = pathlib.Path(save_dir) if save_dir else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
    best_sharpe = -float("inf")

    history = {
        "episode": [], "train_pnl": [],
        "eval_episode": [], "eval_pnl": [], "eval_sharpe": [], "eval_inv": [],
        "epsilon": [],
    }

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)                 # training seeds: 0 .. n_episodes-1
        done = False
        info: dict = {}
        while not done:
            a = agent.act_explore(obs)
            obs2, r, terminated, truncated, info = env.step(a)
            done = terminated or truncated
            agent.buffer.push(obs, a, r, obs2, done)
            obs = obs2
            agent.total_steps += 1
            if agent.total_steps % cfg.train_freq == 0:
                agent.learn_step()
            agent.maybe_sync_target()

        history["episode"].append(ep)
        history["train_pnl"].append(info.get("wealth", 0.0))
        history["epsilon"].append(agent.epsilon())

        if (ep + 1) % eval_every == 0 or ep == 0:
            res = evaluate(env, agent, n_episodes=eval_episodes, seed0=1_000_000)
            history["eval_episode"].append(ep + 1)
            history["eval_pnl"].append(res["pnl_mean"])
            history["eval_sharpe"].append(res["sharpe"])
            history["eval_inv"].append(res["mean_abs_inventory"])
            if verbose:
                marker = ""
                if out is not None:
                    (out / "history.json").write_text(json.dumps(history))
                    if res["sharpe"] > best_sharpe:
                        best_sharpe = res["sharpe"]
                        agent.save(str(out / "dqn.pt"))
                        marker = "  <- saved best"
                print(
                    f"ep {ep + 1:>5}/{n_episodes} | eps {agent.epsilon():.2f} | "
                    f"eval PnL {res['pnl_mean']:7.3f} | Sharpe {res['sharpe']:6.3f} | "
                    f"mean|inv| {res['mean_abs_inventory']:6.2f}{marker}"
                )
            elif out is not None:
                (out / "history.json").write_text(json.dumps(history))
                if res["sharpe"] > best_sharpe:
                    best_sharpe = res["sharpe"]
                    agent.save(str(out / "dqn.pt"))

    return agent, history
