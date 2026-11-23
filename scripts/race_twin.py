"""
Carreras de 4 robots en el gemelo con la lógica EXACTA de despliegue de Webots:
políticas .npz ejecutadas con numpy, muestreo de π (PPO/A2C/DAC) o ε-greedy 0.05
(DQN), escudo de seguridad final (con regla de desbloqueo). 20 carreras por
experimento rotando la parrilla. Guarda results/experiments/<E>/races_final.json
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from race_core.policy_numpy import NumpyPolicy
from race_core.sensing import N_ACTIONS
from rl.train import race
from run_experiments import EXPERIMENTS, OUT

ALG = ["DAC", "PPO", "DQN", "A2C"]


class Deployed:
    def __init__(self, path, seed):
        self.p = NumpyPolicy(path); self.rng = np.random.RandomState(seed)

    def act(self, obs, greedy=False):
        pr = self.p.prefs(obs)
        if self.p.kind == "q":
            a = pr.argmax(1)
            r = self.rng.rand(len(a)) < 0.05
            a = np.where(r, self.rng.randint(0, N_ACTIONS, len(a)), a)
        else:
            c = pr.cumsum(1); u = self.rng.rand(len(pr), 1)
            a = np.minimum((u > c).sum(1), N_ACTIONS - 1)
        return a, pr


if __name__ == "__main__":
    names = sys.argv[1:] or sorted(EXPERIMENTS)
    for e in names:
        d = os.path.join(OUT, e)
        if not os.path.exists(os.path.join(d, "summary.json")):
            continue
        ex = EXPERIMENTS[e]
        agents = {a: Deployed(os.path.join(d, f"{a}.npz"), 100 + k) for k, a in enumerate(ALG)}
        res, traj = race(agents, ex["track"], n_races=20, seed=7000, use_shield=ex.get("shield", True), record=True)
        out = dict(races=res, traj0=[dict(pos=np.round(t["pos"], 3).tolist()) for t in traj])
        json.dump(out, open(os.path.join(d, "races_final.json"), "w"))
        w = {a: sum(r["winner"] == a for r in res) for a in ALG}
        fin = {a: np.mean([a in r["order"] for r in res]) for a in ALG}
        print(e, "victorias", w, "llegan", {a: round(v, 2) for a, v in fin.items()}, flush=True)
