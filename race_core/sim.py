"""
Simulador cinemático vectorizado de la pista (gemelo rápido de Webots).

Reproduce, con los parámetros calibrados en Webots, la marcha del DARwIn-OP
controlada por el GaitManager: velocidades por acción, retardo de primer orden
de la marcha, ruido de deriva, zonas de piso, caídas y colisiones.
Se usa para entrenar (decenas de miles de episodios no son viables en la
física completa de Webots) y para correr cientos de carreras de evaluación.

RaceWorld maneja N instancias en paralelo, cada una con M robots:
  * robots "controlados" (acciones externas: agentes RL)
  * robots "fantasma" (controlador de seguimiento de carril) durante entrenamiento
"""
import math
import numpy as np
from .track import Track, ROBOT_RADIUS, HUMAN_RADIUS, project
from .sensing import (ACT_V, ACT_W, DT, V_MAX, W_MAX, N_ACTIONS, OBS_DIM, STOP,
                      build_obs, static_clearance, zone_flags)
from . import safety

R_GOAL = 20.0
R_COLLISION = -10.0
R_FALL = -3.0
R_SHIELD = -0.5
R_TIME = -0.05
FALL_STEPS = 12          # pasos (s) que tarda en levantarse (MotionManager pág. 10/11)


class RaceWorld:
    def __init__(self, track: Track, num_envs=8, n_robots=1, controlled=None,
                 use_shield=True, collision_mode="terminate", seed=0,
                 noise=1.0, ghost_speed=(0.6, 1.0), random_slot=True):
        self.tr = track
        self.N, self.M = num_envs, n_robots
        self.ctrl = np.array(controlled if controlled is not None else [True] + [False] * (n_robots - 1))
        self.use_shield = use_shield
        self.collision_mode = collision_mode
        self.rng = np.random.RandomState(seed)
        self.noise = noise
        self.ghost_speed = ghost_speed
        self.random_slot = random_slot
        self.max_steps = track.max_steps(DT, V_MAX)
        self.H = len(track.pedestrians)
        if self.H:
            self.ped_a = np.array([p["a"] for p in track.pedestrians])
            self.ped_b = np.array([p["b"] for p in track.pedestrians])
            self.ped_len = np.linalg.norm(self.ped_b - self.ped_a, axis=1)
            self.ped_speed = np.array([p["speed"] for p in track.pedestrians])
            self.ped_pause = np.array([p.get("pause", 0.0) for p in track.pedestrians])
        N, M = self.N, self.M
        self.pos = np.zeros((N, M, 2)); self.yaw = np.zeros((N, M))
        self.v = np.zeros((N, M)); self.w = np.zeros((N, M))
        self.idx = np.zeros((N, M), int); self.s = np.zeros((N, M))
        self.fall_t = np.zeros((N, M), int)
        self.finished = np.zeros((N, M), bool)
        self.crashed = np.zeros((N, M), bool)
        self.finish_time = np.full((N, M), np.nan)
        self.t = np.zeros(N, int)
        self.ped_phase = np.zeros((N, max(self.H, 1)))
        self.g_lat = np.zeros((N, M)); self.g_spd = np.ones((N, M)); self.g_delay = np.zeros((N, M), int)
        self.slots = np.zeros((N, M), int)
        self.stats = dict(collisions=np.zeros((N, M)), falls=np.zeros((N, M)),
                          shield=np.zeros((N, M)), cost=np.zeros((N, M)), ret=np.zeros((N, M)))
        for i in range(N):
            self._reset_env(i)

    # ------------------------------------------------------------------
    def _reset_env(self, i, slots=None):
        grid = self.tr.grid_positions(4)
        M = self.M
        if slots is None:
            slots = self.rng.permutation(4)[:M] if self.random_slot else np.arange(M) % 4
        self.slots[i] = slots
        for j in range(M):
            x, y, yaw = grid[slots[j] % 4]
            back = (slots[j] // 4) * 0.4
            self.pos[i, j] = [x - back * math.cos(yaw), y - back * math.sin(yaw)]
            self.pos[i, j] += self.rng.normal(0, 0.02, 2) * self.noise
            self.yaw[i, j] = yaw + self.rng.normal(0, 0.05) * self.noise
        self.v[i] = 0; self.w[i] = 0; self.fall_t[i] = 0
        self.finished[i] = False; self.crashed[i] = False; self.finish_time[i] = np.nan
        idx, s, _ = project(self.tr, self.pos[i])
        self.idx[i] = idx; self.s[i] = s
        self.t[i] = 0
        if self.H:
            cyc = 2 * (self.ped_len / self.ped_speed + self.ped_pause)
            self.ped_phase[i, :self.H] = self.rng.uniform(0, 1, self.H) * cyc
        self.g_lat[i] = self.rng.uniform(-0.25, 0.25, M)
        self.g_spd[i] = self.rng.uniform(*self.ghost_speed, M)
        self.g_delay[i] = self.rng.randint(0, 6, M)
        for k in self.stats:
            self.stats[k][i] = 0

    def reset(self):
        for i in range(self.N):
            self._reset_env(i)
        return self.observe()

    # ------------------------------------------------------------------
    def humans(self, phase=None):
        """Posiciones de peatones (N,H,2) — movimiento de vaivén."""
        if not self.H:
            return np.zeros((self.N, 0, 2))
        ph = self.ped_phase if phase is None else phase
        # fase en segundos: cruzar (len/speed) -> esperar (pause) -> volver -> esperar
        Tc = (self.ped_len / self.ped_speed)[None]; Tp = self.ped_pause[None]
        cyc = 2 * (Tc + Tp)
        u = ph[:, :self.H] % cyc
        tri = np.where(u < Tc, u / Tc, np.where(u < Tc + Tp, 1.0,
                       np.where(u < 2 * Tc + Tp, 1.0 - (u - Tc - Tp) / Tc, 0.0)))
        return self.ped_a[None] + (self.ped_b - self.ped_a)[None] * tri[..., None]

    def _active(self):
        return ~(self.finished | self.crashed)

    def others(self, j):
        """posiciones de los demás robots activos vistos por el robot j (N,M-1,2);
        los inactivos se mandan lejos."""
        idx = [k for k in range(self.M) if k != j]
        if not idx:
            return np.zeros((self.N, 0, 2))
        P = self.pos[:, idx].copy()
        inactive = ~self._active()[:, idx]
        P[inactive] = 1e3
        return P

    def observe(self, j=None):
        js = range(self.M) if j is None else [j]
        out = []
        H = self.humans()
        for k in js:
            o, _ = build_obs(self.tr, self.pos[:, k], self.yaw[:, k], self.v[:, k], self.w[:, k],
                             self.idx[:, k], H, self.others(k))
            out.append(o)
        return np.stack(out, 1)          # (N, M or 1, OBS)

    # ------------------------------------------------------------------
    def ghost_actions(self):
        """Controlador de seguimiento de carril para robots no controlados."""
        N, M = self.N, self.M
        acts = np.full((N, M), STOP)
        nC = len(self.tr.center)
        for j in range(M):
            j_ = np.clip(self.idx[:, j] + 12, 0, nC - 1)
            lat = self.tr.free_lat[j_] + 0.3 * self.g_lat[:, j]
            tgt = self.tr.center[j_] + self.tr.normal[j_] * lat[:, None]
            d = tgt - self.pos[:, j]
            err = (np.arctan2(d[:, 1], d[:, 0]) - self.yaw[:, j] + np.pi) % (2 * np.pi) - np.pi
            a = np.where(err > 0.5, 6, np.where(err < -0.5, 7,
                         np.where(err > 0.15, 2, np.where(err < -0.15, 3, 0))))
            a = np.where(self.g_spd[:, j] < 0.8, np.where(a == 0, 1, a), a)
            a = np.where(self.t < self.g_delay[:, j], STOP, a)
            acts[:, j] = a
        return acts

    # ------------------------------------------------------------------
    def step(self, actions, prefs=None):
        """actions (N,M) para todos los robots (los fantasmas se ignoran y se
        calculan internamente). prefs (N,M,A) opcional para el escudo.
        Devuelve rewards (N,M), costs (N,M), done_env (N,), info."""
        N, M = self.N, self.M
        tr = self.tr
        ga = self.ghost_actions()
        acts = np.where(self.ctrl[None, :], actions, ga).astype(int)
        proposed = acts.copy()
        H_now = self.humans()
        next_phase = self.ped_phase + DT
        if self.H:
            # el peatón cede el paso: no avanza si chocaría con algún robot
            Hc = self.humans(next_phase)                       # (N,H,2)
            act_now = self._active()
            d = np.linalg.norm(Hc[:, :, None, :] - self.pos[:, None, :, :], axis=-1)  # (N,H,M)
            d = np.where(act_now[:, None, :], d, 9.0)
            moved = np.linalg.norm(Hc - H_now, axis=-1) > 1e-6
            block = (d.min(-1) < HUMAN_RADIUS + ROBOT_RADIUS + 0.12) & moved
            next_phase = next_phase.copy()
            next_phase[:, :self.H] = np.where(block, self.ped_phase[:, :self.H], next_phase[:, :self.H])
        H_next = self.humans(next_phase)
        shielded = np.zeros((N, M), bool)
        if self.use_shield:
            for j in range(M):
                if not self.ctrl[j]:
                    continue
                pr = None if prefs is None else prefs[:, j]
                a2, interv, _ = safety.shield(tr, acts[:, j], self.pos[:, j], self.yaw[:, j],
                                              self.v[:, j], self.w[:, j], H_next, self.others(j), pr)
                act_mask = self._active()[:, j] & (self.fall_t[:, j] == 0)
                shielded[:, j] = interv & act_mask
                acts[:, j] = np.where(act_mask, a2, acts[:, j])

        active = self._active()
        frozen = self.fall_t > 0
        av = ACT_V[acts]; aw = ACT_W[acts]
        slip, rough = zone_flags(tr, self.s.reshape(-1))
        slip = slip.reshape(N, M); rough = rough.reshape(N, M)
        mult_v = np.where(slip > 0, 0.8, 1.0) * np.where(rough > 0, 0.7, 1.0)
        v_new = self.v + safety.LAG_V * (av * mult_v - self.v)
        w_new = self.w + safety.LAG_W * (aw * np.where(slip > 0, 1.15, 1.0) - self.w)
        # ruido de deriva de la marcha (mayor en piso resbaladizo)
        walking = ((np.abs(av) > 0) | (np.abs(aw) > 0)).astype(float)   # sin deriva si está quieto
        sig = self.noise * (1.0 + 2.0 * slip) * walking
        v_new = v_new + self.rng.normal(0, 0.004, (N, M)) * sig
        w_new = w_new + self.rng.normal(0, 0.03, (N, M)) * sig
        v_new = np.where(frozen | ~active, 0, v_new); w_new = np.where(frozen | ~active, 0, w_new)

        # ---------------- caída por maniobra brusca ----------------
        dv = np.abs(av - self.v) / V_MAX + np.abs(aw - self.w) / W_MAX
        p_fall = 0.0005 + slip * 0.03 * np.maximum(dv - 0.6, 0) + 0.01 * np.maximum(dv - 1.4, 0)
        fell = (self.rng.rand(N, M) < p_fall) & active & ~frozen

        # ---------------- integración con detección de colisión ----------------
        collided = np.zeros((N, M), bool)
        pos = self.pos.copy(); yaw = self.yaw.copy()
        n_sub = 4
        for k in range(n_sub):
            yaw_mid = yaw + 0.5 * w_new * DT / n_sub
            newp = pos + np.stack([np.cos(yaw_mid), np.sin(yaw_mid)], -1) * (v_new * DT / n_sub)[..., None]
            newyaw = yaw + w_new * DT / n_sub
            moving = active & ~frozen & ~collided
            for j in range(M):
                st = static_clearance(tr, newp[:, j]) - ROBOT_RADIUS
                hit = st < 0
                if self.H:
                    hd = np.sqrt(((newp[:, j][:, None] - H_now) ** 2).sum(-1)).min(1) - HUMAN_RADIUS - ROBOT_RADIUS
                    hit |= hd < 0
                for q in range(M):
                    if q == j:
                        continue
                    dq = np.linalg.norm(newp[:, j] - pos[:, q], axis=1) - 2 * ROBOT_RADIUS
                    hit |= (dq < 0) & active[:, q]
                hit &= moving[:, j]
                collided[:, j] |= hit
                upd = moving[:, j] & ~hit
                pos[upd, j] = newp[upd, j]; yaw[upd, j] = newyaw[upd, j]
        self.pos, self.yaw = pos, (yaw + np.pi) % (2 * np.pi) - np.pi
        self.v = np.where(collided, 0, v_new); self.w = np.where(collided, 0, w_new)
        # una colisión hace caer al humanoide
        fell |= collided
        self.fall_t = np.where(frozen, self.fall_t - 1, self.fall_t)
        self.fall_t = np.where(fell & active, FALL_STEPS, self.fall_t)

        # ---------------- progreso ----------------
        s_prev = self.s.copy()
        for j in range(M):
            idx, s, _ = project(tr, self.pos[:, j], self.idx[:, j], window=40)
            self.idx[:, j] = idx; self.s[:, j] = s
        self.ped_phase = next_phase
        self.t += 1

        H_new = self.humans()
        st = np.zeros((N, M)); hc = np.zeros((N, M)); rc = np.zeros((N, M))
        for j in range(M):
            st[:, j], hc[:, j], rc[:, j] = safety.clearance(tr, self.pos[:, j], H_new, self.others(j))
        cost = safety.cost(st, hc, rc, collided, fell & ~collided) * active

        reward = (self.s - s_prev) / (V_MAX * DT) + R_TIME
        reward = reward + R_SHIELD * shielded + R_FALL * (fell & ~collided) + R_COLLISION * collided
        newly_fin = active & (self.s >= tr.length - 0.05)
        reward = reward + R_GOAL * newly_fin
        reward = reward * active
        self.finished |= newly_fin
        self.finish_time = np.where(newly_fin, self.t[:, None] * DT, self.finish_time)
        if self.collision_mode == "terminate":
            self.crashed |= collided & active

        st_ = self.stats
        st_["collisions"] += collided; st_["falls"] += fell & ~collided
        st_["shield"] += shielded; st_["cost"] += cost; st_["ret"] += reward

        ctrl_idx = np.where(self.ctrl)[0]
        all_done = (~self._active()[:, ctrl_idx]).all(1)
        timeout = self.t >= self.max_steps
        done = all_done | timeout
        info = dict(proposed=proposed, executed=acts, shielded=shielded, collided=collided,
                    fell=fell, finished=newly_fin, timeout=timeout & ~all_done)
        return reward, cost, done, info


class TrainEnv:
    """Interfaz de entrenamiento: un agente controlado (robot 0) + fantasmas,
    N instancias en paralelo con reinicio automático."""
    def __init__(self, track, num_envs=8, n_ghosts=2, use_shield=True, seed=0, noise=1.0):
        self.world = RaceWorld(track, num_envs, 1 + n_ghosts, use_shield=use_shield,
                               collision_mode="terminate", seed=seed, noise=noise)
        self.num_envs = num_envs
        self.obs_dim, self.n_actions = OBS_DIM, N_ACTIONS
        self.episodes = []          # dicts con métricas por episodio
        self.total_steps = 0

    def reset(self):
        return self.world.reset()[:, 0]

    def step(self, actions, prefs=None):
        w = self.world
        a = np.full((w.N, w.M), STOP); a[:, 0] = actions
        pf = None
        if prefs is not None:
            pf = np.zeros((w.N, w.M, N_ACTIONS)); pf[:, 0] = prefs
        r, c, done, info = w.step(a, pf)
        self.total_steps += w.N
        r, c = r[:, 0], c[:, 0]
        term = done.copy()
        for i in np.where(done)[0]:
            self.episodes.append(dict(
                gstep=self.total_steps, ret=float(w.stats["ret"][i, 0]), cost=float(w.stats["cost"][i, 0]),
                success=bool(w.finished[i, 0]), collisions=int(w.stats["collisions"][i, 0]),
                falls=int(w.stats["falls"][i, 0]), shield=int(w.stats["shield"][i, 0]),
                steps=int(w.t[i]), time=float(w.t[i] * DT),
                progress=float(w.s[i, 0] / w.tr.length)))
            w._reset_env(i)
        obs = w.observe(0)[:, 0]
        info["timeout_env"] = info["timeout"]
        return obs, r, c, term, info
