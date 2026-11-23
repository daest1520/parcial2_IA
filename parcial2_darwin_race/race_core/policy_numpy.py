"""Ejecución de políticas exportadas (.npz) sólo con numpy (para Webots)."""
import numpy as np


class NumpyPolicy:
    def __init__(self, path):
        d = np.load(path, allow_pickle=False)
        n = int(d["n_layers"])
        self.W = [d[f"W{i}"] for i in range(n)]
        self.b = [d[f"b{i}"] for i in range(n)]
        self.act_fn = str(d["act"])
        self.kind = str(d["kind"])         # "logits" (PPO/A2C/DAC) o "q" (DQN)
        self.meta = str(d["meta"]) if "meta" in d else ""

    def forward(self, obs):
        x = np.asarray(obs, np.float32)
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            x = x @ W.T + b
            if i < len(self.W) - 1:
                x = np.tanh(x) if self.act_fn == "tanh" else np.maximum(x, 0)
        return x

    def prefs(self, obs):
        """Preferencias normalizadas (probabilidades) por acción."""
        z = self.forward(obs)
        if self.kind == "q":
            z = z * 5.0
        z = z - z.max(-1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(-1, keepdims=True)

    def act(self, obs, greedy=True, rng=np.random):
        p = self.prefs(obs)
        if greedy:
            return p.argmax(-1), p
        c = p.cumsum(-1)
        u = rng.rand(len(p), 1)
        return (u > c).sum(-1).clip(0, p.shape[-1] - 1), p
