"""Proximal Policy Optimization market maker, implemented from scratch.

Where DQN learns a *value* and acts greedily, PPO learns a *policy* directly: a
neural network that outputs a probability distribution over actions, improved by
gradient ascent on expected reward. This suits market making because the natural
action - how far to place each quote - is **continuous**, and a stochastic
policy explores it smoothly rather than over a coarse discrete grid.

The pieces, each defensible:

* **Actor-critic.** One network with two heads: an *actor* that outputs the mean
  of a Gaussian over the 2-D action (bid, ask offsets; the std is a learned
  state-independent parameter), and a *critic* that estimates the state value
  ``V(s)`` used as a variance-reducing baseline.

* **Generalized Advantage Estimation (GAE).** The advantage ``A(s,a)`` - how
  much better an action did than the critic expected - is estimated with an
  exponentially-weighted sum of temporal-difference errors, trading bias against
  variance via ``lambda``.

* **Clipped surrogate objective.** PPO maximises
  ``min(r*A, clip(r, 1-eps, 1+eps)*A)`` where ``r`` is the ratio of new to old
  action probability. The clip removes the incentive to move the policy too far
  in one update - the trust-region idea that makes PPO stable - letting us do
  several optimisation epochs over each batch of experience.

* **Entropy bonus** keeps the policy from collapsing to a deterministic one too
  early, preserving exploration.

The agent exposes the same ``act(obs)`` interface as the others (returning the
deterministic mean action for evaluation), so it drops straight into the shared
evaluation harness.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


class ActorCritic(nn.Module):
    """Shared-body actor-critic for a continuous (Gaussian) policy."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: tuple[int, ...] = (128, 128)):
        super().__init__()
        layers: list[nn.Module] = []
        last = obs_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.Tanh()]
            last = h
        self.body = nn.Sequential(*layers)
        self.mean_head = nn.Linear(last, act_dim)
        self.value_head = nn.Linear(last, 1)
        # State-independent log-std (a common, stable choice for continuous PPO).
        self.log_std = nn.Parameter(torch.zeros(act_dim) - 0.5)

    def forward(self, x: torch.Tensor):
        h = self.body(x)
        mean = self.mean_head(h)
        value = self.value_head(h).squeeze(-1)
        return mean, value

    def distribution(self, x: torch.Tensor) -> torch.distributions.Normal:
        mean, _ = self.forward(x)
        std = self.log_std.exp().expand_as(mean)
        return torch.distributions.Normal(mean, std)


@dataclass
class PPOConfig:
    hidden: tuple[int, ...] = (128, 128)
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    n_epochs: int = 10            # optimisation passes over each rollout
    minibatch_size: int = 256
    vf_coef: float = 0.5          # value-loss weight
    ent_coef: float = 0.003       # entropy-bonus weight
    grad_clip: float = 0.5
    seed: int = 0


class PPO:
    """Holds the actor-critic; ``act`` is the deterministic eval policy."""

    def __init__(self, obs_dim: int, act_dim: int, cfg: PPOConfig | None = None):
        self.cfg = cfg or PPOConfig()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        torch.manual_seed(self.cfg.seed)
        self.net = ActorCritic(obs_dim, act_dim, self.cfg.hidden)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.cfg.lr)

    def act(self, obs: np.ndarray) -> np.ndarray:
        """Deterministic action (distribution mean), clipped to the action range."""
        with torch.no_grad():
            mean, _ = self.net(torch.from_numpy(np.asarray(obs, dtype=np.float32)))
        return np.clip(mean.numpy(), -1.0, 1.0)

    def reset(self) -> None:
        pass

    def save(self, path: str) -> None:
        torch.save(
            {"net": self.net.state_dict(), "cfg": self.cfg,
             "obs_dim": self.obs_dim, "act_dim": self.act_dim},
            path,
        )

    @classmethod
    def load(cls, path: str) -> "PPO":
        ckpt = torch.load(path, weights_only=False)
        agent = cls(ckpt["obs_dim"], ckpt["act_dim"], ckpt["cfg"])
        agent.net.load_state_dict(ckpt["net"])
        return agent


def _compute_gae(rewards, values, dones, last_value, gamma, lam):
    """Generalized Advantage Estimation; returns (advantages, returns)."""
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        next_value = last_value if t == T - 1 else values[t + 1]
        next_nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        last_gae = delta + gamma * lam * next_nonterminal * last_gae
        adv[t] = last_gae
    returns = adv + values
    return adv, returns


def train_ppo(
    env,
    cfg: PPOConfig | None = None,
    n_iterations: int = 300,
    steps_per_iter: int = 4_000,
    eval_every: int = 20,
    eval_episodes: int = 60,
    verbose: bool = True,
    save_dir: str | None = None,
):
    """Train a PPO agent on a continuous-action ``env``; returns ``(agent, history)``.

    Like the DQN trainer, the best-by-Sharpe model and history are checkpointed
    to ``save_dir`` so interrupted runs leave usable artifacts.
    """
    import json
    import pathlib

    from ..eval import evaluate

    cfg = cfg or PPOConfig()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    agent = PPO(obs_dim, act_dim, cfg)
    rng = np.random.default_rng(cfg.seed)

    out = pathlib.Path(save_dir) if save_dir else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
    best_sharpe = -float("inf")

    history = {"iter": [], "eval_pnl": [], "eval_sharpe": [], "eval_inv": []}

    obs, _ = env.reset(seed=int(rng.integers(1 << 30)))
    for it in range(n_iterations):
        # ---- collect a rollout ----
        b_obs = np.zeros((steps_per_iter, obs_dim), dtype=np.float32)
        b_act = np.zeros((steps_per_iter, act_dim), dtype=np.float32)
        b_logp = np.zeros(steps_per_iter, dtype=np.float32)
        b_rew = np.zeros(steps_per_iter, dtype=np.float32)
        b_done = np.zeros(steps_per_iter, dtype=np.float32)
        b_val = np.zeros(steps_per_iter, dtype=np.float32)

        for t in range(steps_per_iter):
            obs_t = torch.from_numpy(obs)
            with torch.no_grad():
                dist = agent.net.distribution(obs_t)
                action = dist.sample()
                logp = dist.log_prob(action).sum(-1)
                _, value = agent.net(obs_t)
            clipped = torch.clamp(action, -1.0, 1.0).numpy()
            next_obs, reward, terminated, truncated, _ = env.step(clipped)
            done = terminated or truncated

            b_obs[t] = obs
            b_act[t] = action.numpy()
            b_logp[t] = float(logp)
            b_rew[t] = reward
            b_done[t] = float(done)
            b_val[t] = float(value)

            obs = next_obs
            if done:
                obs, _ = env.reset(seed=int(rng.integers(1 << 30)))

        with torch.no_grad():
            _, last_value = agent.net(torch.from_numpy(obs))
        adv, returns = _compute_gae(
            b_rew, b_val, b_done, float(last_value), cfg.gamma, cfg.gae_lambda
        )
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)  # normalise advantages

        # ---- PPO update: several epochs of minibatch SGD ----
        obs_t = torch.from_numpy(b_obs)
        act_t = torch.from_numpy(b_act)
        old_logp_t = torch.from_numpy(b_logp)
        adv_t = torch.from_numpy(adv)
        ret_t = torch.from_numpy(returns)
        idx = np.arange(steps_per_iter)

        for _ in range(cfg.n_epochs):
            rng.shuffle(idx)
            for start in range(0, steps_per_iter, cfg.minibatch_size):
                mb = idx[start:start + cfg.minibatch_size]
                dist = agent.net.distribution(obs_t[mb])
                new_logp = dist.log_prob(act_t[mb]).sum(-1)
                _, value = agent.net(obs_t[mb])

                ratio = torch.exp(new_logp - old_logp_t[mb])
                surr1 = ratio * adv_t[mb]
                surr2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv_t[mb]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = (ret_t[mb] - value).pow(2).mean()
                entropy = dist.entropy().sum(-1).mean()
                loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy

                agent.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.net.parameters(), cfg.grad_clip)
                agent.opt.step()

        # ---- periodic out-of-sample evaluation ----
        if (it + 1) % eval_every == 0 or it == 0:
            res = evaluate(env, agent, n_episodes=eval_episodes, seed0=1_000_000)
            history["iter"].append(it + 1)
            history["eval_pnl"].append(res["pnl_mean"])
            history["eval_sharpe"].append(res["sharpe"])
            history["eval_inv"].append(res["mean_abs_inventory"])
            if out is not None:
                (out / "history.json").write_text(json.dumps(history))
                if res["sharpe"] > best_sharpe:
                    best_sharpe = res["sharpe"]
                    agent.save(str(out / "ppo.pt"))
            if verbose:
                print(
                    f"iter {it + 1:>4}/{n_iterations} | eval PnL {res['pnl_mean']:7.3f} | "
                    f"Sharpe {res['sharpe']:6.3f} | mean|inv| {res['mean_abs_inventory']:6.2f}"
                )

    return agent, history
