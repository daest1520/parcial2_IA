"""
Agente base 3 — Actor-Critic (A2C síncrono, Mnih et al. 2016).
Un actor π_θ(a|s) y un crítico V_φ(s) sobre un tronco compartido, ventaja GAE(λ),
bonificación de entropía. Implementación propia en PyTorch (misma red y
hiperparámetros base que el DAC para que la comparación sea justa).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..common import DEVICE, ortho_init, mlp, compute_gae, export_mlp

DEFAULTS = dict(n_steps=32, lr=7e-4, gamma=0.99, lam=0.95, ent_coef=0.01, vf_coef=0.5,
                max_grad_norm=0.5, reward_scale=0.1, hidden=(128, 128))


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=(128, 128)):
        super().__init__()
        self.encoder = mlp(obs_dim, hidden)
        self.actor = nn.Linear(hidden[-1], n_actions)
        self.critic = nn.Linear(hidden[-1], 1)
        self.apply(ortho_init); ortho_init(self.actor, 0.01); ortho_init(self.critic, 1.0)

    def forward(self, obs):
        h = self.encoder(obs)
        return self.actor(h), self.critic(h).squeeze(1)


def train(env, total_steps, cfg, lagr, log_fn=None, seed=0):
    c = dict(DEFAULTS); c.update(cfg)
    torch.manual_seed(seed); np.random.seed(seed)
    N, T, A, D = env.num_envs, c["n_steps"], env.n_actions, env.obs_dim
    net = ActorCritic(D, A, c["hidden"]).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=c["lr"], eps=1e-5)
    b_obs = torch.zeros((T, N, D)); b_act = torch.zeros((T, N), dtype=torch.long)
    b_rew = torch.zeros((T, N)); b_cost = torch.zeros((T, N)); b_done = torch.zeros((T, N))
    obs = torch.as_tensor(env.reset())
    n_updates = total_steps // (N * T); gstep = 0; diag = []
    for upd in range(1, n_updates + 1):
        for g in opt.param_groups:
            g["lr"] = c["lr"] * (1 - (upd - 1) / n_updates)
        for t in range(T):
            with torch.no_grad():
                logits, _ = net(obs)
                p = F.softmax(logits, -1)
                action = torch.distributions.Categorical(probs=p).sample()
            nobs, r, cst, done, info = env.step(action.numpy(), prefs=p.numpy())
            b_obs[t] = obs; b_act[t] = action
            b_rew[t] = torch.as_tensor(r, dtype=torch.float32); b_cost[t] = torch.as_tensor(cst, dtype=torch.float32)
            b_done[t] = torch.as_tensor(done, dtype=torch.float32)
            obs = torch.as_tensor(nobs)
        gstep += N * T
        lam_ = lagr.update(b_cost.mean().item())
        rew = (b_rew - lam_ * b_cost) * c["reward_scale"]
        flat_obs = b_obs.reshape(-1, D); flat_act = b_act.reshape(-1)
        with torch.no_grad():
            _, v_all = net(flat_obs); _, v_last = net(obs)
            adv, ret = compute_gae(rew, v_all.view(T, N), b_done, v_last, c["gamma"], c["lam"])
        logits, v = net(flat_obs)
        logp = F.log_softmax(logits, -1)
        logp_a = logp.gather(1, flat_act[:, None]).squeeze(1)
        a_flat = adv.reshape(-1); a_flat = (a_flat - a_flat.mean()) / (a_flat.std() + 1e-8)
        pg_loss = -(a_flat * logp_a).mean()
        entropy = -(logp.exp() * logp).sum(1).mean()
        v_loss = ((v - ret.reshape(-1)) ** 2).mean()
        loss = pg_loss - c["ent_coef"] * entropy + c["vf_coef"] * v_loss
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), c["max_grad_norm"]); opt.step()
        if upd % 20 == 0:
            diag.append(dict(step=gstep, entropy=entropy.item(), lam=lam_))
            if log_fn:
                log_fn(gstep, dict(entropy=entropy.item(), lam=lam_))
    return Agent(net), diag


class Agent:
    def __init__(self, net):
        self.net = net

    @torch.no_grad()
    def act(self, obs, greedy=False):
        logits, _ = self.net(torch.as_tensor(obs, dtype=torch.float32))
        p = F.softmax(logits, -1)
        a = p.argmax(-1) if greedy else torch.distributions.Categorical(probs=p).sample()
        return a.numpy(), p.numpy()

    def export(self, path, meta=None):
        export_mlp(self.net.encoder, self.net.actor, "tanh", path, "logits", meta)
