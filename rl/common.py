import math
import numpy as np
import torch
import torch.nn as nn

DEVICE = torch.device("cpu")


class RunningMeanStd:
    """Media/varianza móviles (algoritmo paralelo de Chan) — igual que en el Taller 2."""
    def __init__(self, shape=()):
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = 1e-4

    def update(self, x):
        x = np.asarray(x, np.float64)
        b_mean, b_var, b_count = x.mean(0), x.var(0), x.shape[0]
        delta = b_mean - self.mean
        tot = self.count + b_count
        self.mean = self.mean + delta * b_count / tot
        m2 = self.var * self.count + b_var * b_count + delta ** 2 * self.count * b_count / tot
        self.var = m2 / tot
        self.count = tot


def ortho_init(m, gain=math.sqrt(2)):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain)
        nn.init.zeros_(m.bias)


def mlp(inp, hidden, act=nn.Tanh):
    layers, d = [], inp
    for h in hidden:
        layers += [nn.Linear(d, h), act()]
        d = h
    return nn.Sequential(*layers)


def compute_gae(rewards, values, dones, last_value, gamma, lam):
    """rewards, values, dones: (T, N). dones[t]=1 si el episodio terminó en t."""
    T = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    gae = torch.zeros_like(last_value)
    for t in reversed(range(T)):
        nv = last_value if t == T - 1 else values[t + 1]
        nonterm = 1.0 - dones[t]
        delta = rewards[t] + gamma * nv * nonterm - values[t]
        gae = delta + gamma * lam * nonterm * gae
        adv[t] = gae
    return adv, adv + values


def export_mlp(seq_layers, head, act, path, kind, meta=None):
    """Exporta una política MLP a .npz (para ejecutarla en Webots sólo con numpy)."""
    arrs = {}
    i = 0
    for m in list(seq_layers) + [head]:
        if isinstance(m, nn.Linear):
            arrs[f"W{i}"] = m.weight.detach().cpu().numpy().astype(np.float32)
            arrs[f"b{i}"] = m.bias.detach().cpu().numpy().astype(np.float32)
            i += 1
    arrs["n_layers"] = np.array(i)
    arrs["act"] = np.array(act)
    arrs["kind"] = np.array(kind)
    if meta:
        arrs["meta"] = np.array(str(meta))
    np.savez(path, **arrs)
