"""Arnés común: entrenamiento, evaluación y carreras de los 4 agentes."""
import os, sys, json, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from race_core.track import Track
from race_core.sim import TrainEnv, RaceWorld
from race_core.sensing import N_ACTIONS, STOP, DT
from race_core.safety import Lagrangian
from rl.algos import ppo, dqn, a2c, dac

torch.set_num_threads(1)
ALGOS = {"DAC": dac, "PPO": ppo, "DQN": dqn, "A2C": a2c}


def make_track(tcfg):
    return Track(**tcfg)


def train_agent(algo, tcfg, total_steps, hp=None, seed=0, num_envs=16, n_ghosts=2,
                use_shield=True, use_lagrangian=True, verbose=False):
    tr = make_track(tcfg)
    env = TrainEnv(tr, num_envs=num_envs, n_ghosts=n_ghosts, use_shield=use_shield, seed=seed)
    lagr = Lagrangian(enabled=use_lagrangian)
    t0 = time.time()
    log = (lambda s, d: print(algo, s, {k: round(v, 3) for k, v in d.items()}, flush=True)) if verbose else None
    agent, diag = ALGOS[algo].train(env, total_steps, hp or {}, lagr, log_fn=log, seed=seed)
    return agent, dict(episodes=env.episodes, diag=diag, train_time=time.time() - t0, lam=lagr.lam)


@torch.no_grad()
def evaluate(agent, tcfg, episodes=32, seed=1000, n_ghosts=2, use_shield=True, greedy=False):
    """Episodios de prueba (solo el agente + fantasmas), con semillas nuevas."""
    tr = make_track(tcfg)
    env = TrainEnv(tr, num_envs=episodes, n_ghosts=n_ghosts, use_shield=use_shield, seed=seed)
    obs = env.reset()
    done_mask = np.zeros(episodes, bool)
    w = env.world
    res = [None] * episodes
    while not done_mask.all():
        a, p = agent.act(obs, greedy=greedy)
        acts = np.full((w.N, w.M), STOP); acts[:, 0] = a
        pf = np.zeros((w.N, w.M, N_ACTIONS)); pf[:, 0] = p
        r, c, done, info = w.step(acts, pf)
        for i in np.where(done & ~done_mask)[0]:
            res[i] = dict(ret=float(w.stats["ret"][i, 0]), cost=float(w.stats["cost"][i, 0]),
                          success=bool(w.finished[i, 0]), time=float(w.finish_time[i, 0]) if w.finished[i, 0] else np.nan,
                          collisions=int(w.stats["collisions"][i, 0]), falls=int(w.stats["falls"][i, 0]),
                          shield=int(w.stats["shield"][i, 0]), progress=float(min(1, w.s[i, 0] / tr.length)),
                          steps=int(w.t[i]))
            done_mask[i] = True
        # congelar entornos terminados (no reiniciar)
        w.crashed |= done_mask[:, None]
        obs = w.observe(0)[:, 0]
    return res


@torch.no_grad()
def race(agents, tcfg, n_races=20, seed=5000, use_shield=True, greedy=False, record=False):
    """Carreras: los 4 agentes en la misma pista. Cada carrera rota la parrilla.
    Las colisiones derriban al robot (se levanta en FALL_STEPS s) y la carrera sigue."""
    names = list(agents)
    tr = make_track(tcfg)
    M = len(names)
    world = RaceWorld(tr, num_envs=n_races, n_robots=M, controlled=[True] * M, use_shield=use_shield,
                      collision_mode="fall", seed=seed, random_slot=False)
    for i in range(n_races):           # rotación de la parrilla de salida
        world._reset_env(i, slots=np.roll(np.arange(M), i))
    traj = [] if record else None
    done_all = np.zeros(n_races, bool)
    while not done_all.all():
        obs = world.observe()
        acts = np.zeros((n_races, M), int); pf = np.zeros((n_races, M, N_ACTIONS))
        for j, nm in enumerate(names):
            a, p = agents[nm].act(obs[:, j], greedy=greedy)
            acts[:, j] = a; pf[:, j] = p
        if record:
            traj.append(dict(pos=world.pos[0].copy(), yaw=world.yaw[0].copy(), hum=world.humans()[0].copy()))
        r, c, done, info = world.step(acts, pf)
        done_all |= done
        world.crashed |= done_all[:, None] & ~world.finished   # congelar carreras terminadas
    out = []
    for i in range(n_races):
        ft = world.finish_time[i]
        order = [names[j] for j in np.argsort(np.where(np.isnan(ft), 1e9, ft)) if not np.isnan(ft[j])]
        out.append(dict(winner=order[0] if order else None, order=order, slots=world.slots[i].tolist(),
                        finish_time={names[j]: (None if np.isnan(ft[j]) else float(ft[j])) for j in range(M)},
                        progress={names[j]: float(min(1, world.s[i, j] / tr.length)) for j in range(M)},
                        collisions={names[j]: int(world.stats["collisions"][i, j]) for j in range(M)},
                        falls={names[j]: int(world.stats["falls"][i, j]) for j in range(M)},
                        shield={names[j]: int(world.stats["shield"][i, j]) for j in range(M)},
                        ret={names[j]: float(world.stats["ret"][i, j]) for j in range(M)}))
    return out, traj
