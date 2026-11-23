"""
race_brain — supervisor de la carrera (Webots R2025a, Python).

Cada DT = 1 s:
  1. lee la pose de los 4 DARwIn-OP y de los peatones (modo supervisor),
  2. construye la MISMA observación de 27 dimensiones usada en el entrenamiento
     (rayos virtuales, rumbo a la pista, zona de piso, humano y robot más cercanos),
  3. consulta la política de cada agente (DAC, PPO, DQN, A2C; archivos .npz,
     ejecutados sólo con numpy),
  4. aplica el ESCUDO DE SEGURIDAD (safe learning) y
  5. envía el comando de marcha "CMD x a" al controlador darwin_walker del robot.

Además mide el desempeño: tiempo de llegada, orden, progreso, colisiones,
caídas, intervenciones del escudo; muestra un marcador en pantalla y guarda
results/<mundo>_<fecha>.json y la trayectoria en .csv.

Argumentos (controllerArgs):
  --track=../../tracks/carrera_S.json   pista (generada por scripts/make_world.py)
  --policies=../../policies             carpeta con <ALGO>.npz (o <mundo>/<ALGO>.npz)
  --mode=race|explore                   explore: muestreo con mayor temperatura
  --max-time=900                        tiempo límite de carrera [s]
  --grid=0                              rotación de la parrilla de salida
  --no-shield                           desactiva el escudo (sólo para ablación)
  --quit                                cierra Webots al terminar (modo batch)
"""
import os, sys, math, json, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
import numpy as np
from controller import Supervisor
from race_core.track import Track, project, ROBOT_RADIUS, HUMAN_RADIUS
from race_core.sensing import build_obs, ACT_X, ACT_A, ACT_NAMES, DT, N_ACTIONS, STOP
from race_core import safety
from race_core.policy_numpy import NumpyPolicy

AGENTS = ["DAC", "PPO", "DQN", "A2C"]
COLORS = {"DAC": 0xD91A1A, "PPO": 0x1A59E6, "DQN": 0x1AB333, "A2C": 0xF2A60D}


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="../../tracks/carrera_S.json")
    ap.add_argument("--policies", default="../../policies")
    ap.add_argument("--mode", default="race")
    ap.add_argument("--max-time", type=float, default=900.0)
    ap.add_argument("--grid", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-shield", action="store_true")
    ap.add_argument("--quit", action="store_true")
    ap.add_argument("--out", default="../../../results/webots")
    a = ap.parse_args()
    # variables de entorno para ejecución por lotes (scripts/webots_batch.sh)
    if os.environ.get("RACE_GRID"): a.grid = int(os.environ["RACE_GRID"]); a.seed = a.grid
    if os.environ.get("RACE_MAXTIME"): a.max_time = float(os.environ["RACE_MAXTIME"])
    if os.environ.get("RACE_OUT"): a.out = os.environ["RACE_OUT"]
    if os.environ.get("RACE_QUIT") == "1": a.quit = True
    if os.environ.get("RACE_NO_SHIELD") == "1": a.no_shield = True
    return a


def axis_angle(yaw, pitch):
    """orientación de Viewpoint (eje-ángulo) para mirar con rumbo yaw e inclinación pitch (>0 hacia abajo)."""
    cz, sz, cy, sy = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]); Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    R = Rz @ Ry
    ang = math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2)))
    if ang < 1e-6:
        return [0, 0, 1, 0]
    ax = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(ang))
    return [float(ax[0]), float(ax[1]), float(ax[2]), ang]


def rel(p):
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(HERE, p))


def main():
    a = parse()
    rng = np.random.RandomState(a.seed)
    sup = Supervisor()
    ts = int(sup.getBasicTimeStep())
    track_path = rel(a.track)
    world = os.path.splitext(os.path.basename(track_path))[0]
    with open(track_path) as fh:
        tj = json.load(fh)
    tr = Track(**tj["cfg"])
    print(f"[race_brain] pista {world}: {tr.size} {tr.Lx}x{tr.Ly} m, L={tr.length:.2f} m", flush=True)

    pol_dir = rel(a.policies)
    pols = {}
    for nm in AGENTS:
        for cand in (os.path.join(pol_dir, world, f"{nm}.npz"), os.path.join(pol_dir, f"{nm}.npz")):
            if os.path.exists(cand):
                pols[nm] = NumpyPolicy(cand)
                print(f"[race_brain] {nm}: política {os.path.relpath(cand, pol_dir)} ({pols[nm].kind})", flush=True)
                break
        else:
            raise FileNotFoundError(f"No hay política para {nm} en {pol_dir}")

    robots = [sup.getFromDef(f"RACER_{nm}") for nm in AGENTS]
    humans = []
    k = 0
    while sup.getFromDef(f"HUMAN_{k}") is not None:
        humans.append(sup.getFromDef(f"HUMAN_{k}")); k += 1
    em = sup.getDevice("emitter")
    rx = sup.getDevice("receiver"); rx.enable(ts)
    M, H = len(AGENTS), len(humans)

    # parrilla de salida (rotación opcional, para alternar posiciones entre carreras)
    grid = tr.grid_positions(4)
    for j, node in enumerate(robots):
        x, y, yaw = grid[(j + a.grid) % 4]
        node.getField("translation").setSFVec3f([x, y, 0.236])
        node.getField("rotation").setSFRotation([0, 0, 1, yaw])
        node.resetPhysics()

    def send(j, msg):
        em.setChannel(j + 1)
        em.send(msg.encode())

    def poses():
        P = np.zeros((M, 2)); Y = np.zeros(M); up = np.ones(M, bool)
        for j, n in enumerate(robots):
            p = n.getPosition(); R = n.getOrientation()
            P[j] = p[:2]; Y[j] = math.atan2(R[3], R[0])
            up[j] = R[8] > 0.6          # eje z del torso casi vertical
        return P, Y, up

    def human_pos():
        return np.array([h.getPosition()[:2] for h in humans]).reshape(1, H, 2)

    # --- espera inicial: los robots adoptan la postura y arranca la marcha ---
    t_wait = sup.getTime() + 3.0
    while sup.getTime() < t_wait:
        for j in range(M):
            send(j, "STOP")
        if sup.step(ts) == -1:
            return
    P, Y, up = poses()
    idx, s, _ = project(tr, P)
    v = np.zeros(M); w = np.zeros(M)
    finished = np.zeros(M, bool); fin_time = np.full(M, np.nan)
    fallen = np.zeros(M, bool); fall_t = np.zeros(M)
    stats = {k: np.zeros(M, int) for k in ("collisions", "falls", "shield_interventions", "decisions")}
    in_contact = np.zeros(M, bool)
    traj = []
    t0 = sup.getTime(); next_t = t0
    prev_P, prev_Y = P.copy(), Y.copy()
    temp = 1.5 if a.mode == "explore" else 1.0
    print("[race_brain] ¡SALIDA!", flush=True)

    while sup.step(ts) != -1:
        t = sup.getTime() - t0
        # ---------- mensajes de caída del darwin_walker ----------
        while rx.getQueueLength() > 0:
            msg = rx.getString() if hasattr(rx, "getString") else rx.getData().decode()
            rx.nextPacket()
            parts = msg.strip("\x00").split()
            if len(parts) == 2 and parts[1] in AGENTS:
                j = AGENTS.index(parts[1])
                if parts[0] == "FALL" and not fallen[j]:
                    fallen[j] = True; stats["falls"][j] += 1; fall_t[j] = t
                elif parts[0] == "UP":
                    fallen[j] = False
        P, Y, up = poses()
        # ---------- colisiones (contacto físico) ----------
        st = tr.sdf_query(P) - ROBOT_RADIUS
        hd = (np.sqrt(((P[:, None] - human_pos()[0][None]) ** 2).sum(-1)).min(1) - HUMAN_RADIUS - ROBOT_RADIUS) if H else np.full(M, 9.0)
        rd = np.full(M, 9.0)
        for j in range(M):
            for q in range(M):
                if q != j and not finished[q] and not finished[j]:
                    rd[j] = min(rd[j], np.linalg.norm(P[j] - P[q]) - 2 * ROBOT_RADIUS)
        # histéresis: un contacto empieza con penetración > 2 cm y termina con holgura > 2 cm
        start_c = ((st < -0.02) | (hd < -0.02) | (rd < -0.02)) & ~finished
        end_c = (st > 0.02) & (hd > 0.02) & (rd > 0.02)
        stats["collisions"] += (start_c & ~in_contact).astype(int)
        in_contact = (in_contact | start_c) & ~end_c
        for j in range(M):
            if not up[j] and not fallen[j]:
                fallen[j] = True; stats["falls"][j] += 1; fall_t[j] = t
            elif up[j] and fallen[j] and t - fall_t[j] > 15.0:
                fallen[j] = False        # se levantó (por si se perdió el mensaje "UP")
        # ---------- decisión cada DT ----------
        if sup.getTime() < next_t:
            continue
        dt_real = max(sup.getTime() - (next_t - DT), 1e-3) if t > 0 else DT
        next_t += DT
        d = P - prev_P
        fwd = d[:, 0] * np.cos(Y) + d[:, 1] * np.sin(Y)
        v = np.clip(fwd / DT, -0.02, 0.12)
        w = np.clip(((Y - prev_Y + np.pi) % (2 * np.pi) - np.pi) / DT, -1.2, 1.2)
        prev_P, prev_Y = P.copy(), Y.copy()
        idx, s, lat = project(tr, P, idx, window=40)
        # llegada
        for j in range(M):
            if not finished[j] and s[j] >= tr.length - 0.05:
                finished[j] = True; fin_time[j] = t
                send(j, "STOP")
                pos = len([x for x in fin_time if not np.isnan(x)])
                print(f"[race_brain] {AGENTS[j]} cruza la meta en {t:.1f} s (puesto {pos})", flush=True)
        Hn = human_pos()
        others = np.zeros((M, M - 1, 2))
        for j in range(M):
            oth = [q for q in range(M) if q != j]
            others[j] = np.where(finished[oth][:, None], 1e3, P[oth])
        Hm = np.repeat(Hn, M, 0)
        obs, rays = build_obs(tr, P, Y, v, w, idx, Hm, others)
        acts = np.full(M, STOP); prefs = np.zeros((M, N_ACTIONS))
        for j, nm in enumerate(AGENTS):
            p = pols[nm].prefs(obs[j:j + 1])[0]
            if temp != 1.0:
                p = p ** (1.0 / temp); p /= p.sum()
            if pols[nm].kind == "q":           # DQN: ε-greedy con ε=0.05
                acts[j] = rng.randint(N_ACTIONS) if rng.rand() < 0.05 else int(p.argmax())
            else:                              # PPO/A2C/DAC: muestreo de π(a|s)
                acts[j] = int(rng.choice(N_ACTIONS, p=p))
            prefs[j] = p
        proposed = acts.copy(); shielded = np.zeros(M, bool)
        if not a.no_shield:
            acts, shielded, _ = safety.shield(tr, acts, P, Y, v, w, Hm, others, prefs)
        for j in range(M):
            if finished[j]:
                continue
            if fallen[j]:
                send(j, "STOP"); continue
            stats["decisions"][j] += 1
            stats["shield_interventions"][j] += int(shielded[j])
            send(j, f"CMD {ACT_X[acts[j]]:.2f} {ACT_A[acts[j]]:.2f}")
            traj.append([round(t, 2), AGENTS[j], round(P[j, 0], 3), round(P[j, 1], 3), round(Y[j], 3),
                         round(float(s[j]), 3), ACT_NAMES[proposed[j]], ACT_NAMES[acts[j]], int(shielded[j]),
                         round(float(v[j]), 3), round(float(w[j]), 3)])
        # ---------- marcador ----------
        order = sorted(range(M), key=lambda j: (-1e9 + fin_time[j]) if finished[j] else -s[j])
        sup.setLabel(0, f"CARRERA DARwIn-OP  t = {t:6.1f} s   pista {world}", 0.45, 0.01, 0.07, 0xFFFFFF, 0, "Arial")
        for r, j in enumerate(order):
            st_txt = f"META {fin_time[j]:.1f}s" if finished[j] else f"{100 * s[j] / tr.length:5.1f}%"
            if fallen[j]:
                st_txt += " (caído)"
            sup.setLabel(1 + r, f"{r + 1}. {AGENTS[j]:<4} {st_txt}  col {stats['collisions'][j]}  caídas {stats['falls'][j]}  escudo {stats['shield_interventions'][j]}",
                         0.45, 0.06 + 0.04 * r, 0.06, COLORS[AGENTS[j]], 0, "Arial")
        snap = os.environ.get("RACE_SNAPSHOT")
        if snap and int(t) in (2, 3, 30, 31, 90, 91, 160, 161, 240, 241) and abs(t - round(t)) < 0.05:
            vp = sup.getFromDef("VIEW")
            if vp is not None:
                if int(t) % 2 == 0:     # vista cenital de toda la pista
                    vp.getField("position").setSFVec3f([tr.Lx / 2, tr.Ly / 2 - 1.0, max(tr.Lx, tr.Ly) * 1.6])
                    vp.getField("orientation").setSFRotation(axis_angle(math.pi / 2, 1.38))
                else:                   # vista de persecución del líder
                    jl = int(np.argmax(np.where(finished, -1, s)))
                    vp.getField("position").setSFVec3f([P[jl, 0] - 1.2 * math.cos(Y[jl]), P[jl, 1] - 1.2 * math.sin(Y[jl]), 0.9])
                    vp.getField("orientation").setSFRotation(axis_angle(Y[jl], 0.55))
            sup.step(ts)
            sup.exportImage(f"{snap}_{int(t):03d}.jpg", 90)
        # ---------- cuadros para video (Colab / modo sin ventana) ----------
        frames = os.environ.get("RACE_FRAMES")
        every = int(os.environ.get("RACE_FRAME_EVERY", "2"))
        if frames and int(round(t)) % every == 0:
            vp = sup.getFromDef("VIEW")
            if vp is not None and not getattr(main, "_view_set", False):
                # cenital: altura suficiente para que quepa toda la pista (vista ancha 16:9)
                h = max(0.5 * (tr.Lx + 0.6) / math.tan(0.39), 0.5 * (tr.Ly + 1.4) / 0.21)
                vp.getField("position").setSFVec3f([tr.Lx / 2, tr.Ly / 2 + 0.2, h])
                vp.getField("orientation").setSFRotation(axis_angle(math.pi / 2, math.pi / 2 - 1e-3))
                main._view_set = True
                sup.step(ts)
            os.makedirs(frames, exist_ok=True)
            sup.exportImage(os.path.join(frames, f"f_{int(round(t)):05d}.jpg"), 85)
        if finished.all() or t >= a.max_time:
            break

    for j in range(M):
        send(j, "STOP")
    order = [AGENTS[j] for j in np.argsort(np.where(np.isnan(fin_time), 1e9, fin_time)) if not np.isnan(fin_time[j])]
    res = dict(world=world, track=tj["cfg"], length=tr.length, grid_rotation=a.grid, mode=a.mode,
               shield=not a.no_shield, winner=order[0] if order else None, order=order,
               finish_time={AGENTS[j]: (None if np.isnan(fin_time[j]) else float(fin_time[j])) for j in range(M)},
               progress={AGENTS[j]: (1.0 if finished[j] else float(min(1.0, s[j] / tr.length))) for j in range(M)},
               **{k: {AGENTS[j]: int(v_[j]) for j in range(M)} for k, v_ in stats.items()})
    out = rel(a.out); os.makedirs(out, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    base = os.path.join(out, f"{world}_g{a.grid}_{stamp}")
    with open(base + ".json", "w") as fh:
        json.dump(res, fh, indent=1)
    with open(base + "_trayectoria.csv", "w") as fh:
        fh.write("t,robot,x,y,yaw,s,accion_propuesta,accion_ejecutada,escudo,v,w\n")
        for row in traj:
            fh.write(",".join(str(x) for x in row) + "\n")
    print("[race_brain] RESULTADO " + json.dumps(res), flush=True)
    if a.quit:
        sup.simulationQuit(0)
    else:
        while sup.step(ts) != -1:
            pass


if __name__ == "__main__":
    main()
