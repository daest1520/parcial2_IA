"""
Sensores virtuales, modelo de acciones y observación (comunes a entrenamiento y Webots).

Acciones discretas de alto nivel -> comando de marcha (v, w) que el
GaitManager del DARwIn-OP traduce en pasos.
"""
import math
import numpy as np
from .track import ROBOT_RADIUS, HUMAN_RADIUS, project

DT = 1.0                    # s por decisión del agente (~1.7 pasos de marcha)
V_MAX = 0.084               # m/s  (medido en Webots: GaitManager X=1.0)
W_MAX = 0.90                # rad/s (medido en Webots: GaitManager A=1.0)

# (v [m/s], w [rad/s], amplitud X, amplitud A, nombre)
# v y w fueron CALIBRADOS ejecutando el RobotisOp2GaitManager en Webots R2025a
# (scripts/calibrate_gait: 8 s por comando, tras 3 s de régimen transitorio).
ACTIONS = [
    (0.084, 0.000, 1.0, 0.0, "avanzar_rapido"),
    (0.049, 0.000, 0.5, 0.0, "avanzar_lento"),
    (0.080, 0.286, 1.0, 0.3, "avanzar_izq"),
    (0.080, -0.286, 1.0, -0.3, "avanzar_der"),
    (0.034, 0.482, 0.6, 0.5, "arco_izq"),
    (0.034, -0.482, 0.6, -0.5, "arco_der"),
    (0.000, 0.880, 0.0, 1.0, "girar_izq"),
    (0.000, -0.880, 0.0, -1.0, "girar_der"),
    (0.000, 0.000, 0.0, 0.0, "detenerse"),
]
N_ACTIONS = len(ACTIONS)
ACT_V = np.array([a[0] for a in ACTIONS])
ACT_W = np.array([a[1] for a in ACTIONS])
ACT_X = np.array([a[2] for a in ACTIONS])
ACT_A = np.array([a[3] for a in ACTIONS])
ACT_NAMES = [a[4] for a in ACTIONS]
STOP = N_ACTIONS - 1

N_RAYS = 9
RAY_ANGLES = np.linspace(-math.pi / 2, math.pi / 2, N_RAYS)
RAY_RANGE = 1.5
OBS_DIM = N_RAYS + 2 + 2 + 1 + 1 + 2 + 2 + 4 + 4   # = 27


# ------------------------------------------------------------------------
def raycast(origins, angles, segs, circles, max_range=RAY_RANGE):
    """origins (N,2), angles (N,R) absolutos. segs (S,4); circles (N,C,3) o (C,3).
    Devuelve distancias (N,R)."""
    N, R = angles.shape
    dx, dy = np.cos(angles), np.sin(angles)
    best = np.full((N, R), max_range)
    if len(segs):
        ox = origins[:, 0][:, None, None]; oy = origins[:, 1][:, None, None]
        x1, y1, x2, y2 = [segs[:, k][None, None, :] for k in range(4)]
        ex, ey = x2 - x1, y2 - y1
        ddx, ddy = dx[:, :, None], dy[:, :, None]
        den = ddx * ey - ddy * ex
        with np.errstate(divide="ignore", invalid="ignore"):
            t = ((x1 - ox) * ey - (y1 - oy) * ex) / den
            u = ((x1 - ox) * ddy - (y1 - oy) * ddx) / den
        ok = (np.abs(den) > 1e-12) & (t > 0) & (u >= 0) & (u <= 1)
        t = np.where(ok, t, max_range)
        best = np.minimum(best, t.min(-1))
    if circles is not None and len(circles):
        C = circles if circles.ndim == 3 else np.broadcast_to(circles, (N,) + circles.shape)
        if C.shape[1]:
            cx = C[:, None, :, 0] - origins[:, 0][:, None, None]
            cy = C[:, None, :, 1] - origins[:, 1][:, None, None]
            r = C[:, None, :, 2]
            b = cx * dx[:, :, None] + cy * dy[:, :, None]
            c = cx ** 2 + cy ** 2 - r ** 2
            disc = b * b - c
            sq = np.sqrt(np.clip(disc, 0, None))
            t = b - sq
            t = np.where((disc >= 0) & (t > 0), t, max_range)
            best = np.minimum(best, t.min(-1))
    return np.clip(best, 0, max_range)


def seg_point_dist(p, segs):
    """p (N,2), segs (S,4) -> distancia mínima (N,)"""
    if not len(segs):
        return np.full(len(p), 1e9)
    a = segs[None, :, :2]; b = segs[None, :, 2:]
    ab = b - a
    ap = p[:, None, :] - a
    t = np.clip((ap * ab).sum(-1) / ((ab * ab).sum(-1) + 1e-12), 0, 1)
    q = a + ab * t[..., None]
    return np.sqrt(((p[:, None, :] - q) ** 2).sum(-1)).min(1)


def circle_dist(p, circles):
    """distancia borde a borde centro-robot (N,) a círculos (N,C,3) o (C,3)."""
    if circles is None or circles.shape[-2] == 0:
        return np.full(len(p), 1e9)
    C = circles if circles.ndim == 3 else np.broadcast_to(circles, (len(p),) + circles.shape)
    d = np.sqrt(((p[:, None, :] - C[..., :2]) ** 2).sum(-1)) - C[..., 2]
    return d.min(1)


def static_clearance(track, p):
    return track.sdf_query(p)


def raycast_sdf(track, origins, angles, dyn_circles, max_range=RAY_RANGE, iters=16):
    """Rayos: geometría estática por sphere-tracing sobre el SDF + círculos
    dinámicos (humanos/robots) analíticos."""
    dx, dy = np.cos(angles), np.sin(angles)
    t = np.zeros(angles.shape)
    o = origins[:, None, :]
    hit = np.zeros(angles.shape, bool)
    ox, oy = origins[:, 0:1], origins[:, 1:2]
    P = np.empty(angles.shape + (2,))
    for _ in range(iters):
        P[..., 0] = ox + dx * t; P[..., 1] = oy + dy * t
        d = track.sdf_query(P, nearest=True)
        hit |= d < 0.015
        t = np.where(hit, t, np.minimum(t + np.maximum(d, 0.015), max_range))
    if dyn_circles is not None and dyn_circles.shape[1]:
        t = np.minimum(t, raycast(origins, angles, np.zeros((0, 4)), dyn_circles, max_range))
    return np.clip(t, 0, max_range)


def zone_flags(track, s):
    slip = np.zeros_like(s); rough = np.zeros_like(s)
    for kind, a, b in track.zone_ranges:
        inside = (s >= a) & (s <= b)
        if kind == "slippery":
            slip = np.maximum(slip, inside)
        else:
            rough = np.maximum(rough, inside)
    return slip, rough


def _rel(p, yaw, q):
    """vector relativo de q respecto a (p,yaw): distancia y bearing"""
    d = q - p
    dist = np.sqrt((d ** 2).sum(-1))
    ang = np.arctan2(d[..., 1], d[..., 0]) - yaw
    return dist, ang


def build_obs(track, pos, yaw, v, w, idx, humans, robots):
    """
    pos (N,2), yaw (N,), v,w (N,), idx (N,) índice en la línea central.
    humans: (N,H,2) posiciones de peatones; robots: (N,K,2) otros robots.
    """
    N = len(pos)
    dyn = []
    if humans.shape[1]:
        dyn.append(np.concatenate([humans, np.full(humans.shape[:2] + (1,), HUMAN_RADIUS)], -1))
    if robots.shape[1]:
        dyn.append(np.concatenate([robots, np.full(robots.shape[:2] + (1,), ROBOT_RADIUS)], -1))
    dyn = np.concatenate(dyn, 1) if dyn else np.zeros((N, 0, 3))
    rays = raycast_sdf(track, pos, yaw[:, None] + RAY_ANGLES[None, :], dyn)
    o = [rays / RAY_RANGE]
    nC = len(track.center)
    for look in (12, 30):          # 0.6 m y 1.5 m adelante (puntos cada 5 cm)
        j = np.clip(idx + look, 0, nC - 1)
        _, ang = _rel(pos, yaw, track.center[j])
        o += [np.sin(ang)[:, None], np.cos(ang)[:, None]]
    lat = ((pos - track.center[idx]) * track.normal[idx]).sum(1)
    o.append((lat / (track.width / 2))[:, None])
    o.append((1.0 - track.s[idx] / track.length)[:, None])
    o.append((v / V_MAX)[:, None]); o.append((w / W_MAX)[:, None])
    slip, rough = zone_flags(track, track.s[idx])
    o += [slip[:, None], rough[:, None]]
    for group in (humans, robots):
        if group.shape[1]:
            dist, ang = _rel(pos[:, None, :], yaw[:, None], group)
            k = dist.argmin(1)
            dd = dist[np.arange(N), k]; aa = ang[np.arange(N), k]
            present = (dd < 2.0).astype(float)
            o += [present[:, None], (np.minimum(dd, 2.0) / 2.0)[:, None],
                  (np.sin(aa) * present)[:, None], (np.cos(aa) * present)[:, None]]
        else:
            o += [np.zeros((N, 1)), np.ones((N, 1)), np.zeros((N, 1)), np.zeros((N, 1))]
    return np.concatenate(o, 1).astype(np.float32), rays
