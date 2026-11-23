"""Reúne tablas para el informe -> JSON (lo consume report/make_report.js)."""
import os, sys, json, glob
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from race_core.track import Track
from race_core.sensing import ACTIONS
from run_experiments import EXPERIMENTS
from rl.algos import dac, ppo, dqn, a2c

ALG = ["DAC", "PPO", "DQN", "A2C"]
NAMES = {"avanzar_rapido": "Avanzar rápido", "avanzar_lento": "Avanzar lento", "avanzar_izq": "Avanzar + izq.",
         "avanzar_der": "Avanzar + der.", "arco_izq": "Arco izq.", "arco_der": "Arco der.",
         "girar_izq": "Girar izq.", "girar_der": "Girar der.", "detenerse": "Detenerse"}


def main(text_path, out):
    D = dict(root=ROOT)
    D["tracks"] = []
    for sz in "SML":
        t = Track(size=sz)
        D["tracks"].append(dict(size=sz, area=f"{t.Lx}×{t.Ly}", length=t.length, obst=len(t.obstacles), ped=len(t.pedestrians)))
    D["actions"] = [[NAMES[a[4]], f"{a[2]:.1f}", f"{a[3]:+.1f}" if a[3] else "0", f"{a[0]:.3f}", f"{a[1]:+.2f}" if a[1] else "0"] for a in ACTIONS]
    da, pp, dq, ac = dac.DEFAULTS, ppo.DEFAULTS, dqn.DEFAULTS, a2c.DEFAULTS
    D["params"] = [
        ["Red (capas ocultas)", "128-128 tanh, tronco común", "64-64 tanh, π y V separadas", "128-128 ReLU", "128-128 tanh, tronco común"],
        ["Tasa de aprendizaje", f"{da['lr']:.0e}", f"{pp['lr']:.0e}", f"{dq['lr']:.1e}", f"{ac['lr']:.0e}"],
        ["γ / λ GAE", "0.99 / 0.95", "0.99 / 0.95", "0.99 / —", "0.99 / 0.95"],
        ["Lote", f"16×{da['n_steps']}", f"16×{pp['n_steps']}, 4 épocas, 4 minilotes", f"{dq['batch']}, memoria 10^5^", f"16×{ac['n_steps']}"],
        ["Exploración", "entropía 0.01 + descubrimiento", "entropía 0.01", "ε 1→0.05 (40%)", "entropía 0.01"],
        ["Específicos", "K=3, M=5, κ=1, c~I~=0.1, η 1→0.1, ε~ask~=0.05", "clip ε=0.2", "objetivo c/2000 pasos, Huber, 2 act./paso", "—"],
        ["Seguridad", "escudo + λ", "escudo + λ", "escudo + λ", "escudo + λ"],
    ]
    fix = lambda t: t.replace("ε_final", "ε~final~").replace("c_I", "c~I~").replace("ε_ask", "ε~ask~")
    D["design"] = [[e, x["factor"], fix(x["desc"])] for e, x in EXPERIMENTS.items()]
    ev = pd.read_csv(os.path.join(ROOT, "results/tables/evaluacion_por_experimento.csv"))
    rc = pd.read_csv(os.path.join(ROOT, "results/tables/carreras.csv"))
    g = ev.groupby("algo"); r = rc.groupby("algo")
    def row(name, fn, fmt):
        return [name] + [fmt(fn(a)) for a in ALG]
    D["global_table"] = [
        row("Tasa de éxito", lambda a: g.success.mean()[a], lambda x: f"{100 * x:.1f}%"),
        row("Recompensa acumulada", lambda a: g.ret.mean()[a], lambda x: f"{x:.1f}"),
        row("Progreso medio", lambda a: g.progress.mean()[a], lambda x: f"{100 * x:.0f}%"),
        row("Tiempo de llegada (s)", lambda a: ev[ev.algo == a].time.mean(), lambda x: f"{x:.0f}"),
        row("Colisiones / episodio", lambda a: g.collisions.mean()[a], lambda x: f"{x:.3f}"),
        row("Caídas / episodio", lambda a: g.falls.mean()[a], lambda x: f"{x:.2f}"),
        row("Intervenciones escudo", lambda a: g.shield.mean()[a], lambda x: f"{x:.0f}"),
        row("Costo acumulado", lambda a: g.cost.mean()[a], lambda x: f"{x:.1f}"),
        row("Victorias en carrera", lambda a: r.win.sum()[a], lambda x: f"{int(x)}"),
        row("Llegan a la meta (carrera)", lambda a: r.finished.mean()[a], lambda x: f"{100 * x:.0f}%"),
        row("Tiempo de entrenamiento (s)", lambda a: g.train_time.mean()[a], lambda x: f"{x:.0f}"),
    ]
    rows = []
    for e in sorted(ev.exp.unique()):
        s = ev[ev.exp == e].set_index("algo")
        best = s.success.max()
        rows.append([e] + [(f"**{100 * s.success[a]:.0f}**" if s.success[a] == best else f"{100 * s.success[a]:.0f}") + f" / {s.ret[a]:.0f}" for a in ALG])
    D["per_exp"] = rows
    # carreras en Webots
    wt = []
    for f in sorted(glob.glob(os.path.join(ROOT, "results/webots/*.json"))):
        w = json.load(open(f))
        ft = w["finish_time"]
        win = w["winner"] or "—"
        wt.append([f"{w['world']} / g{w['grid_rotation']}", win, f"{ft[win]:.0f} s" if w["winner"] else "—",
                   ", ".join(w["order"]) if w["order"] else "ninguno",
                   "/".join(str(w["collisions"][a]) for a in ALG)])
    D["webots_table"] = wt or [["—", "—", "—", "—", "—"]]
    D["text"] = json.load(open(text_path))
    D["summary"] = {}
    json.dump(D, open(out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
