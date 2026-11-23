"""
Agente base 2 — DQN (Mnih et al. 2015).
Q-learning con red neuronal, memoria de repetición (replay buffer), red objetivo
actualizada periódicamente, exploración ε-greedy decreciente y pérdida de Huber.
Implementación propia en PyTorch.

Safe learning: el costo c_t se guarda en la memoria y la recompensa usada en el
objetivo es r_t - λ c_t con el λ vigente (método Lagrangiano).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..common import DEVICE, ortho_init, export_mlp

DEFAULTS = dict(lr=2.5e-4, gamma=0.99, buffer=100_000, batch=128, learning_starts=5_000,
                target_every=2_000, eps_start=1.0, eps_end=0.05, eps_frac=0.4,
                grad_steps=2, reward_scale=0.1, hidden=(128, 128), max_grad_norm=10.0,
                double=False)


class QNet(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=(128, 128)):
        super().__init__()
        layers, d = [], obs_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]; d = h
        self.body = nn.Sequential(*layers)
        self.q = nn.Linear(d, n_actions)
        self.apply(ortho_init)

    def forward(self, x):
        return self.q(self.body(x))


def train(env, total_steps, cfg, lagr, log_fn=None, seed=0):
    c = dict(DEFAULTS); c.update(cfg)
    torch.manual_seed(seed); rng = np.random.RandomState(seed)
    N, A, D = env.num_envs, env.n_actions, env.obs_dim
    q = QNet(D, A, c["hidden"]).to(DEVICE); qt = QNet(D, A, c["hidden"]).to(DEVICE)
    qt.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=c["lr"])
    cap = c["buffer"]
    B_o = np.zeros((cap, D), np.float32); B_n = np.zeros((cap, D), np.float32)
    B_a = np.zeros(cap, np.int64); B_r = np.zeros(cap, np.float32); B_c = np.zeros(cap, np.float32)
    B_d = np.zeros(cap, np.float32)
    ptr, size = 0, 0
    obs = env.reset(); gstep = 0; diag = []; last_target = 0
    costs_window = []
    while gstep < total_steps:
        eps = max(c["eps_end"], c["eps_start"] - (c["eps_start"] - c["eps_end"]) * gstep / (c["eps_frac"] * total_steps))
        with torch.no_grad():
            qv = q(torch.as_tensor(obs)).numpy()
        greedy = qv.argmax(1)
        rand = rng.randint(0, A, N)
        explore = rng.rand(N) < eps
        action = np.where(explore, rand, greedy)
        # preferencias para el escudo: Q (o uniforme si la acción fue aleatoria)
        prefs = np.where(explore[:, None], rng.rand(N, A), qv)
        nobs, r, cst, done, info = env.step(action, prefs=prefs)
        # en reinicio automático nobs es ya la obs. inicial del nuevo episodio; como
        # done=1 enmascara el bootstrap, el valor de nobs no afecta el objetivo.
        idx = (ptr + np.arange(N)) % cap
        B_o[idx] = obs; B_n[idx] = nobs; B_a[idx] = action; B_r[idx] = r; B_c[idx] = cst
        B_d[idx] = done.astype(np.float32) * (~info["timeout_env"]).astype(np.float32)
        ptr = (ptr + N) % cap; size = min(size + N, cap)
        obs = nobs; gstep += N
        costs_window.append(cst.mean())
        if len(costs_window) >= 128:
            lagr.update(float(np.mean(costs_window))); costs_window = []
        if gstep >= c["learning_starts"]:
            for _ in range(c["grad_steps"]):
                j = rng.randint(0, size, c["batch"])
                o = torch.as_tensor(B_o[j]); n = torch.as_tensor(B_n[j]); a = torch.as_tensor(B_a[j])
                rr = torch.as_tensor((B_r[j] - lagr.lam * B_c[j]) * c["reward_scale"])
                d = torch.as_tensor(B_d[j])
                with torch.no_grad():
                    if c["double"]:
                        na = q(n).argmax(1, keepdim=True); qn = qt(n).gather(1, na).squeeze(1)
                    else:
                        qn = qt(n).max(1).values
                    y = rr + c["gamma"] * (1 - d) * qn
                qa = q(o).gather(1, a[:, None]).squeeze(1)
                loss = F.smooth_l1_loss(qa, y)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), c["max_grad_norm"]); opt.step()
            if gstep - last_target >= c["target_every"]:
                qt.load_state_dict(q.state_dict()); last_target = gstep
            if (gstep // N) % 200 == 0:
                diag.append(dict(step=gstep, eps=eps, loss=loss.item(), lam=lagr.lam, q_mean=float(qa.detach().mean())))
                if log_fn:
                    log_fn(gstep, dict(eps=eps, loss=loss.item(), lam=lagr.lam))
    return Agent(q), diag


class Agent:
    def __init__(self, net):
        self.net = net

    @torch.no_grad()
    def act(self, obs, greedy=True, eps=0.05):
        """Evaluación estándar de DQN: ε-greedy con ε=0.05 (Mnih et al., 2015).
        greedy=True fuerza argmax puro."""
        qv = self.net(torch.as_tensor(obs, dtype=torch.float32)).numpy()
        a = qv.argmax(1)
        if not greedy and eps > 0:
            r = np.random.rand(len(a)) < eps
            a = np.where(r, np.random.randint(0, qv.shape[1], len(a)), a)
        return a, qv

    def export(self, path, meta=None):
        export_mlp(self.net.body, self.net.q, "relu", path, "q", meta)
