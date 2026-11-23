"""
DAC — Discovery Actor-Critic (algoritmo propio del Taller 2) adaptado a la carrera
del DARwIn-OP: observación vectorial (27 dims) en lugar de imágenes 84x84x4, por
lo que el codificador Nature-CNN se reemplaza por un MLP. El resto del algoritmo
se conserva tal cual:

  * Ensamble de K críticos extrínsecos V^E_k con máscaras bootstrap; su desacuerdo
    σ_V(s) pondera el gradiente del actor: w(s) = 1/(1+κ σ̃_V(s)).
  * Módulo de descubrimiento: codificador aleatorio fijo ψ y ensamble de M
    modelos de dinámica g_m(ψ(s),a) -> ψ(s'). Recompensa intrínseca = varianza
    entre modelos (novedad epistémica) con crítico propio V^I (no episódico).
  * "Preguntar por otra acción": si σ_V(s) > media móvil (o con prob. ε) se actúa
    con μ(a|s) ∝ π(a|s) exp(η U(s,a)); corrección off-policy ρ̄ = min(1, π/b).
"""
import math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..common import DEVICE, RunningMeanStd, ortho_init, mlp, compute_gae, export_mlp


class ActorCriticEnsemble(nn.Module):
    def __init__(self, obs_dim, n_actions, n_critics=3, hidden=(128, 128)):
        super().__init__()
        self.encoder = mlp(obs_dim, hidden)
        h = hidden[-1]
        self.actor = nn.Linear(h, n_actions)
        self.critics = nn.ModuleList([nn.Linear(h, 1) for _ in range(n_critics)])
        self.critic_int = nn.Linear(h, 1)
        self.apply(ortho_init)
        ortho_init(self.actor, 0.01)
        for c in self.critics:
            ortho_init(c, 1.0)
        ortho_init(self.critic_int, 1.0)

    def forward(self, obs):
        h = self.encoder(obs)
        logits = self.actor(h)
        v_e = torch.cat([c(h) for c in self.critics], dim=1)
        v_i = self.critic_int(h).squeeze(1)
        return logits, v_e, v_i


class DiscoveryModule(nn.Module):
    def __init__(self, obs_dim, n_actions, n_models=5, feat=64, hidden=128):
        super().__init__()
        self.n_actions = n_actions
        self.target = nn.Sequential(nn.Linear(obs_dim, 128), nn.LeakyReLU(),
                                    nn.Linear(128, 128), nn.LeakyReLU(), nn.Linear(128, feat))
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.models = nn.ModuleList([
            nn.Sequential(nn.Linear(feat + n_actions, hidden), nn.ReLU(),
                          nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, feat))
            for _ in range(n_models)])

    @torch.no_grad()
    def features(self, obs):
        f = self.target(obs)
        return f / (f.norm(dim=1, keepdim=True) + 1e-8) * math.sqrt(f.shape[1])

    def predict(self, feat, actions):
        a = F.one_hot(actions, self.n_actions).float()
        x = torch.cat([feat, a], dim=1)
        return torch.stack([m(x) for m in self.models], dim=0)

    @torch.no_grad()
    def disagreement(self, feat, actions):
        return self.predict(feat, actions).var(dim=0).mean(dim=1)

    @torch.no_grad()
    def novelty_map(self, feat):
        B, A = feat.shape[0], self.n_actions
        f = feat.unsqueeze(1).expand(B, A, -1).reshape(B * A, -1)
        a = torch.arange(A, device=feat.device).repeat(B)
        return self.disagreement(f, a).view(B, A)


DEFAULTS = dict(n_steps=32, lr=7e-4, gamma=0.99, gamma_i=0.99, lam=0.95,
                ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, reward_scale=0.1,
                n_critics=3, critic_mask_p=0.8, kappa=1.0,
                use_discovery=True, n_models=5, int_coef=0.1, disc_lr=3e-4, disc_mask_p=0.8,
                eta0=1.0, eta_final=0.1, ask_eps=0.05, hidden=(128, 128))


def train(env, total_steps, cfg, lagr, log_fn=None, seed=0):
    c = dict(DEFAULTS); c.update(cfg)
    torch.manual_seed(seed); np.random.seed(seed)
    N, T, A, D = env.num_envs, c["n_steps"], env.n_actions, env.obs_dim
    K = c["n_critics"]
    net = ActorCriticEnsemble(D, A, K, c["hidden"]).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=c["lr"], eps=1e-5)
    disc = disc_opt = None
    if c["use_discovery"]:
        disc = DiscoveryModule(D, A, c["n_models"]).to(DEVICE)
        disc_opt = torch.optim.Adam(disc.models.parameters(), lr=c["disc_lr"])
    int_rms = RunningMeanStd(); int_run = np.zeros(N); sigma_ema = None

    b_obs = torch.zeros((T, N, D)); b_next = torch.zeros((T, N, D))
    b_act = torch.zeros((T, N), dtype=torch.long); b_logb = torch.zeros((T, N))
    b_rew = torch.zeros((T, N)); b_cost = torch.zeros((T, N)); b_done = torch.zeros((T, N))
    b_ask = torch.zeros((T, N))
    obs = torch.as_tensor(env.reset())
    n_updates = total_steps // (N * T)
    gstep = 0
    diag = []
    for upd in range(1, n_updates + 1):
        frac = (upd - 1) / n_updates
        for g in opt.param_groups:
            g["lr"] = c["lr"] * (1 - frac)
        eta = c["eta0"] + (c["eta_final"] - c["eta0"]) * frac
        for t in range(T):
            with torch.no_grad():
                logits, v_e, _ = net(obs)
                logp_pi = F.log_softmax(logits, -1)
                if disc is not None:
                    sigma = v_e.std(1) if K > 1 else torch.zeros(N)
                    sm = sigma.mean().item()
                    sigma_ema = sm if sigma_ema is None else 0.99 * sigma_ema + 0.01 * sm
                    ask = (sigma > sigma_ema) | (torch.rand(N) < c["ask_eps"])
                    U = disc.novelty_map(disc.features(obs))
                    U = (U - U.mean(1, keepdim=True)) / (U.std(1, keepdim=True) + 1e-8)
                    logp_mu = F.log_softmax(logp_pi + eta * U, -1)
                    logp_b = torch.where(ask.unsqueeze(1), logp_mu, logp_pi)
                else:
                    ask = torch.zeros(N, dtype=torch.bool); logp_b = logp_pi
                action = torch.distributions.Categorical(logits=logp_b).sample()
            nobs, r, cst, done, info = env.step(action.numpy(), prefs=logp_b.exp().numpy())
            b_obs[t] = obs; b_next[t] = torch.as_tensor(nobs)   # (con reinicio: se enmascara con done)
            b_act[t] = action; b_logb[t] = logp_b.gather(1, action[:, None]).squeeze(1)
            b_rew[t] = torch.as_tensor(r, dtype=torch.float32)
            b_cost[t] = torch.as_tensor(cst, dtype=torch.float32)
            b_done[t] = torch.as_tensor(done, dtype=torch.float32); b_ask[t] = ask.float()
            obs = torch.as_tensor(nobs)
        gstep += N * T
        lam_ = lagr.update(b_cost.mean().item())
        rew = (b_rew - lam_ * b_cost) * c["reward_scale"]

        flat_obs = b_obs.reshape(-1, D); flat_next = b_next.reshape(-1, D); flat_act = b_act.reshape(-1)
        if disc is not None:
            f_s, f_n = disc.features(flat_obs), disc.features(flat_next)
            r_int = disc.disagreement(f_s, flat_act).view(T, N).numpy()
            rets = []
            for t in range(T):
                int_run = int_run * c["gamma_i"] + r_int[t]
                rets.append(int_run.copy())
            int_rms.update(np.concatenate(rets))
            r_int = torch.as_tensor(r_int / math.sqrt(int_rms.var + 1e-8), dtype=torch.float32)
        else:
            r_int = torch.zeros((T, N))

        with torch.no_grad():
            _, ve_all, vi_all = net(flat_obs)
            ve_mean = ve_all.mean(1).view(T, N); vi_all = vi_all.view(T, N)
            _, ve_last, vi_last = net(obs)
            adv_e, ret_e = compute_gae(rew, ve_mean, b_done, ve_last.mean(1), c["gamma"], c["lam"])
            adv_i, ret_i = compute_gae(r_int, vi_all, torch.zeros_like(b_done), vi_last, c["gamma_i"], c["lam"])
        adv = adv_e + (c["int_coef"] * adv_i if disc is not None else 0.0)

        logits_all, ve_all, vi_all = net(flat_obs)
        logp = F.log_softmax(logits_all, -1)
        logp_a = logp.gather(1, flat_act[:, None]).squeeze(1)
        rho = torch.exp(logp_a.detach() - b_logb.reshape(-1)).clamp(max=1.0)
        if K > 1:
            sig = ve_all.detach().std(1)
            w = 1.0 / (1.0 + c["kappa"] * sig / (sig.mean() + 1e-8))
        else:
            sig = torch.zeros_like(logp_a); w = torch.ones_like(logp_a)
        a_flat = adv.reshape(-1)
        a_flat = (a_flat - a_flat.mean()) / (a_flat.std() + 1e-8)
        pg_loss = -(rho * w * a_flat * logp_a).mean()
        entropy = -(logp.exp() * logp).sum(1).mean()
        mask = (torch.rand_like(ve_all) < c["critic_mask_p"]).float()
        v_loss_e = (((ve_all - ret_e.reshape(-1, 1)) ** 2) * mask).sum() / mask.sum().clamp(min=1)
        v_loss_i = ((vi_all - ret_i.reshape(-1)) ** 2).mean() if disc is not None else torch.tensor(0.0)
        loss = pg_loss - c["ent_coef"] * entropy + c["vf_coef"] * (v_loss_e + v_loss_i)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), c["max_grad_norm"]); opt.step()

        if disc is not None:
            valid = 1.0 - b_done.reshape(-1)
            pred = disc.predict(f_s, flat_act)
            err = ((pred - f_n.unsqueeze(0)) ** 2).mean(2)
            m = (torch.rand_like(err) < c["disc_mask_p"]).float() * valid.unsqueeze(0)
            dl = (err * m).sum() / m.sum().clamp(min=1)
            disc_opt.zero_grad(); dl.backward(); disc_opt.step()
        if upd % 20 == 0:
            diag.append(dict(step=gstep, entropy=entropy.item(), critic_std=float(sig.mean()),
                             r_int=float(r_int.mean()), ask_rate=float(b_ask.mean()),
                             rho=float(rho.mean()), lam=lam_))
        if log_fn and upd % 20 == 0:
            log_fn(gstep, dict(entropy=entropy.item(), lam=lam_, ask=float(b_ask.mean()),
                               sigma=float(sig.mean())))
    return Agent(net), diag


class Agent:
    """Política desplegable del DAC (sólo actor)."""
    def __init__(self, net):
        self.net = net

    @torch.no_grad()
    def act(self, obs, greedy=False):
        logits, _, _ = self.net(torch.as_tensor(obs, dtype=torch.float32))
        p = F.softmax(logits, -1)
        a = p.argmax(-1) if greedy else torch.distributions.Categorical(probs=p).sample()
        return a.numpy(), p.numpy()

    def export(self, path, meta=None):
        export_mlp(self.net.encoder, self.net.actor, "tanh", path, "logits", meta)
