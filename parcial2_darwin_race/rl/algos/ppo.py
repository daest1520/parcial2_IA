"""
Agente base 1 — PPO (Proximal Policy Optimization, Schulman et al. 2017).
Objetivo recortado L^CLIP = E[min(r_t Â_t, clip(r_t,1-ε,1+ε) Â_t)], varias épocas
de minilotes sobre cada lote de experiencia, GAE(λ), redes actor/crítico separadas.
Implementación propia en PyTorch (hiperparámetros estándar de CleanRL/SB3).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..common import DEVICE, ortho_init, mlp, compute_gae, export_mlp

DEFAULTS = dict(n_steps=128, lr=3e-4, gamma=0.99, lam=0.95, clip=0.2, epochs=4, minibatches=4,
                ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, reward_scale=0.1, hidden=(64, 64))


class PPONet(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=(64, 64)):
        super().__init__()
        self.pi_body = mlp(obs_dim, hidden); self.pi = nn.Linear(hidden[-1], n_actions)
        self.v_body = mlp(obs_dim, hidden); self.v = nn.Linear(hidden[-1], 1)
        self.apply(ortho_init); ortho_init(self.pi, 0.01); ortho_init(self.v, 1.0)

    def forward(self, obs):
        return self.pi(self.pi_body(obs)), self.v(self.v_body(obs)).squeeze(1)


def train(env, total_steps, cfg, lagr, log_fn=None, seed=0):
    c = dict(DEFAULTS); c.update(cfg)
    torch.manual_seed(seed); np.random.seed(seed)
    N, T, A, D = env.num_envs, c["n_steps"], env.n_actions, env.obs_dim
    net = PPONet(D, A, c["hidden"]).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=c["lr"], eps=1e-5)
    b_obs = torch.zeros((T, N, D)); b_act = torch.zeros((T, N), dtype=torch.long)
    b_logp = torch.zeros((T, N)); b_rew = torch.zeros((T, N)); b_cost = torch.zeros((T, N))
    b_done = torch.zeros((T, N)); b_val = torch.zeros((T, N))
    obs = torch.as_tensor(env.reset())
    n_updates = max(1, total_steps // (N * T)); gstep = 0; diag = []
    B = N * T; mb = B // c["minibatches"]
    for upd in range(1, n_updates + 1):
        for g in opt.param_groups:
            g["lr"] = c["lr"] * (1 - (upd - 1) / n_updates)
        for t in range(T):
            with torch.no_grad():
                logits, v = net(obs)
                p = F.softmax(logits, -1)
                dist = torch.distributions.Categorical(probs=p)
                action = dist.sample()
            nobs, r, cst, done, info = env.step(action.numpy(), prefs=p.numpy())
            b_obs[t] = obs; b_act[t] = action; b_logp[t] = dist.log_prob(action); b_val[t] = v
            b_rew[t] = torch.as_tensor(r, dtype=torch.float32); b_cost[t] = torch.as_tensor(cst, dtype=torch.float32)
            b_done[t] = torch.as_tensor(done, dtype=torch.float32)
            obs = torch.as_tensor(nobs)
        gstep += B
        lam_ = lagr.update(b_cost.mean().item())
        rew = (b_rew - lam_ * b_cost) * c["reward_scale"]
        with torch.no_grad():
            _, v_last = net(obs)
            adv, ret = compute_gae(rew, b_val, b_done, v_last, c["gamma"], c["lam"])
        f_obs, f_act, f_logp = b_obs.reshape(-1, D), b_act.reshape(-1), b_logp.reshape(-1)
        f_adv, f_ret = adv.reshape(-1), ret.reshape(-1)
        clipfracs = []
        for ep in range(c["epochs"]):
            perm = torch.randperm(B)
            for k in range(0, B, mb):
                i = perm[k:k + mb]
                logits, v = net(f_obs[i])
                logp = F.log_softmax(logits, -1)
                lp = logp.gather(1, f_act[i][:, None]).squeeze(1)
                ratio = torch.exp(lp - f_logp[i])
                a = f_adv[i]; a = (a - a.mean()) / (a.std() + 1e-8)
                pg = -torch.min(ratio * a, ratio.clamp(1 - c["clip"], 1 + c["clip"]) * a).mean()
                ent = -(logp.exp() * logp).sum(1).mean()
                vl = ((v - f_ret[i]) ** 2).mean()
                loss = pg - c["ent_coef"] * ent + c["vf_coef"] * vl
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), c["max_grad_norm"]); opt.step()
                clipfracs.append(((ratio - 1).abs() > c["clip"]).float().mean().item())
        diag.append(dict(step=gstep, entropy=ent.item(), lam=lam_, clipfrac=float(np.mean(clipfracs))))
        if log_fn:
            log_fn(gstep, dict(entropy=ent.item(), lam=lam_))
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
        export_mlp(self.net.pi_body, self.net.pi, "tanh", path, "logits", meta)
