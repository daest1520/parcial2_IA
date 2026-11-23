"""
Ejecuta los 20 experimentos del parcial. Cada experimento entrena los 4 agentes
(DAC, PPO, DQN, A2C) con el mismo presupuesto, los evalúa en episodios nuevos y
los hace competir en carreras de 4 robots. Guarda políticas (.npz), métricas y
curvas en results/experiments/<EXP>/.

  python scripts/run_experiments.py --steps 400000 --workers 2
  python scripts/run_experiments.py --only E01 E02
"""
import os, sys, json, time, argparse, traceback
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "experiments")
ALG = ["DAC", "PPO", "DQN", "A2C"]

# ----------------------------------------------------------------------------
#  Diseño experimental: tamaño del espacio, salida, meta y parámetros
# ----------------------------------------------------------------------------
BASE_T = dict(size="S")
EXPERIMENTS = {
    # --- tamaño del espacio ---
    "E01": dict(desc="Base: pista S (5.0x3.6 m), salida 0, meta 1.0", factor="base", track=dict(size="S")),
    "E02": dict(desc="Tamaño M (6.5x4.4 m)", factor="tamaño", track=dict(size="M")),
    "E03": dict(desc="Tamaño L (8.0x5.2 m)", factor="tamaño", track=dict(size="L")),
    # --- posición inicial ---
    "E04": dict(desc="Salida adelantada al 15% (después de la 1a recta)", factor="salida", track=dict(size="S", start_frac=0.15)),
    "E05": dict(desc="Salida al 30% (después de la zona resbaladiza)", factor="salida", track=dict(size="S", start_frac=0.30)),
    "E06": dict(desc="Salida en el extremo opuesto (sentido inverso), pista S", factor="salida", track=dict(size="S", reverse=True)),
    "E07": dict(desc="Sentido inverso, pista M", factor="salida", track=dict(size="M", reverse=True)),
    # --- posición de la meta ---
    "E08": dict(desc="Meta al 60% del recorrido", factor="meta", track=dict(size="S", goal_frac=0.6)),
    "E09": dict(desc="Meta al 80% del recorrido", factor="meta", track=dict(size="S", goal_frac=0.8)),
    "E10": dict(desc="Pista M, salida 20% y meta 80%", factor="meta", track=dict(size="M", start_frac=0.2, goal_frac=0.8)),
    "E11": dict(desc="Pista L, salida 30% y meta 100%", factor="meta", track=dict(size="L", start_frac=0.3)),
    # --- parámetros de los algoritmos (comunes) ---
    "E12": dict(desc="Tasa de aprendizaje x0.5", factor="parámetros", track=BASE_T, hp="lr_half"),
    "E13": dict(desc="Tasa de aprendizaje x2", factor="parámetros", track=BASE_T, hp="lr_double"),
    "E14": dict(desc="Factor de descuento γ=0.95", factor="parámetros", track=BASE_T, hp=dict(gamma=0.95)),
    "E15": dict(desc="Más exploración (entropía 0.03 / ε_final 0.1, 60%)", factor="parámetros", track=BASE_T, hp="explore"),
    "E16": dict(desc="Redes más grandes (256,256)", factor="parámetros", track=BASE_T, hp=dict(hidden=(256, 256))),
    # --- safe learning (ablaciones) ---
    "E17": dict(desc="Sin restricción Lagrangiana (λ=0), con escudo", factor="safe learning", track=BASE_T, lagrangian=False),
    "E18": dict(desc="Sin escudo ni Lagrangiano (entrenamiento y evaluación)", factor="safe learning", track=BASE_T, shield=False, lagrangian=False),
    # --- parámetros específicos de cada algoritmo ---
    "E19": dict(desc="Específicos A: DAC κ=0 sin descubrimiento | PPO clip 0.1 | DQN doble | A2C n=64", factor="parámetros",
                track=BASE_T, hp=dict(DAC=dict(kappa=0.0, use_discovery=False), PPO=dict(clip=0.1),
                                      DQN=dict(double=True), A2C=dict(n_steps=64))),
    "E20": dict(desc="Específicos B: DAC más descubrimiento (c_I 0.5, ε_ask 0.2) | PPO 10 épocas | DQN target 500 | A2C lr 1e-3", factor="parámetros",
                track=BASE_T, hp=dict(DAC=dict(int_coef=0.5, ask_eps=0.2), PPO=dict(epochs=10),
                                      DQN=dict(target_every=500), A2C=dict(lr=1e-3))),
}


def hp_for(exp, algo):
    from rl.algos import dac, ppo, dqn, a2c
    D = {"DAC": dac, "PPO": ppo, "DQN": dqn, "A2C": a2c}[algo].DEFAULTS
    hp = exp.get("hp")
    if hp is None:
        return {}
    if hp == "lr_half":
        return dict(lr=D["lr"] * 0.5)
    if hp == "lr_double":
        return dict(lr=D["lr"] * 2)
    if hp == "explore":
        return dict(eps_end=0.1, eps_frac=0.6) if algo == "DQN" else dict(ent_coef=0.03)
    if algo in hp:
        return dict(hp[algo])
    if any(k in hp for k in ALG):
        return {}
    return dict(hp)


def run_one(name, steps, seed=0):
    from rl.train import train_agent, evaluate, race
    exp = EXPERIMENTS[name]
    d = os.path.join(OUT, name); os.makedirs(d, exist_ok=True)
    shield = exp.get("shield", True); lagr = exp.get("lagrangian", True)
    agents, summary = {}, dict(name=name, desc=exp["desc"], factor=exp["factor"], track=exp["track"],
                                shield=shield, lagrangian=lagr, steps=steps, algos={})
    t0 = time.time()
    for algo in ALG:
        hp = hp_for(exp, algo)
        ag, info = train_agent(algo, exp["track"], steps, hp=hp, seed=seed,
                               use_shield=shield, use_lagrangian=lagr)
        ag.export(os.path.join(d, f"{algo}.npz"), meta=dict(exp=name, algo=algo, hp=hp, track=exp["track"]))
        ev = evaluate(ag, exp["track"], episodes=48, seed=1000 + seed, use_shield=shield)
        agents[algo] = ag
        eps = info["episodes"]
        tr_eps = [[e["gstep"], round(e["ret"], 2), int(e["success"]), round(e["cost"], 2),
                   e["collisions"], e["falls"], e["shield"], round(e["progress"], 3)] for e in eps]
        summary["algos"][algo] = dict(hp=hp, train_time=info["train_time"], lam=info["lam"],
                                      n_train_episodes=len(eps), eval=ev, diag=info["diag"][::2])
        with open(os.path.join(d, f"{algo}_train_episodes.json"), "w") as f:
            json.dump(tr_eps, f)
        print(f"[{name}] {algo} listo: succ={np.mean([r['success'] for r in ev]):.2f} "
              f"t={time.time() - t0:.0f}s", flush=True)
    races, traj = race(agents, exp["track"], n_races=12, seed=5000 + seed, use_shield=shield, record=True)
    summary["races"] = races
    summary["race0_traj"] = [dict(pos=np.round(t["pos"], 3).tolist(), hum=np.round(t["hum"], 3).tolist()) for t in traj]
    summary["wall_time"] = time.time() - t0
    with open(os.path.join(d, "summary.json"), "w") as f:
        json.dump(summary, f)
    return name


def _safe(args):
    try:
        return run_one(*args)
    except Exception:
        traceback.print_exc()
        return None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400_000)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    names = a.only or list(EXPERIMENTS)
    names = [n for n in names if not os.path.exists(os.path.join(OUT, n, "summary.json"))]
    print("Experimentos pendientes:", names, flush=True)
    with Pool(a.workers) as p:
        for r in p.imap_unordered(_safe, [(n, a.steps) for n in names]):
            print("TERMINADO", r, flush=True)
