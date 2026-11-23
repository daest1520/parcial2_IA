"""
Safe learning: escudo de seguridad (safety shield) + señal de costo.

1) Escudo (Alshiekh et al., 2018): antes de ejecutar la acción propuesta por
   el agente se predice la pose del robot un paso hacia adelante con el modelo
   de marcha calibrado. Si la acción lleva a:
      - chocar con paredes u obstáculos             (margen MARGIN_STATIC),
      - invadir el espacio personal de un humano     (margen MARGIN_HUMAN),
      - chocar con otro robot                        (margen MARGIN_ROBOT),
   se reemplaza por la acción segura más preferida por la política.
   El agente recibe una pequeña penalización cuando el escudo interviene, para
   que aprenda a no proponer acciones peligrosas.

2) Costo c_t (CMDP): cuenta eventos de riesgo (cercanía excesiva, colisión,
   caída). Los algoritmos optimizan r_t - λ c_t con λ ajustado por ascenso dual
   (método Lagrangiano) para mantener E[c] <= COST_LIMIT.
"""
import numpy as np
from .track import ROBOT_RADIUS, HUMAN_RADIUS
from .sensing import ACT_V, ACT_W, DT, N_ACTIONS, STOP, V_MAX, W_MAX, seg_point_dist, circle_dist

MARGIN_STATIC = 0.08      # m libres respecto a paredes/obstáculos
MARGIN_HUMAN = 0.25       # m de "espacio personal" del humano (borde a borde)
MARGIN_ROBOT = 0.08       # m entre robots (borde a borde)
LAG_V, LAG_W = 0.7, 0.8   # respuesta de primer orden de la marcha por paso
COST_LIMIT = 0.02         # costo medio admisible por paso


def predict(pos, yaw, v, w, a_v, a_w, frac=1.0):
    """Pose predicha tras frac*DT aplicando el modelo de marcha (sin ruido)."""
    v1 = v + LAG_V * (a_v - v)
    w1 = w + LAG_W * (a_w - w)
    dt = DT * frac
    yaw_mid = yaw + 0.5 * w1 * dt
    p = pos + np.stack([np.cos(yaw_mid), np.sin(yaw_mid)], -1) * (v1 * dt)[..., None]
    return p, yaw + w1 * dt


def clearance(track, p, humans, robots):
    """p (N,2). humans (N,H,2), robots (N,K,2). Distancias borde a borde."""
    from .sensing import static_clearance
    st = static_clearance(track, p) - ROBOT_RADIUS
    if humans.shape[1]:
        hc = np.sqrt(((p[:, None] - humans) ** 2).sum(-1)).min(1) - HUMAN_RADIUS - ROBOT_RADIUS
    else:
        hc = np.full(len(p), 9.0)
    if robots.shape[1]:
        rc = np.sqrt(((p[:, None] - robots) ** 2).sum(-1)).min(1) - 2 * ROBOT_RADIUS
    else:
        rc = np.full(len(p), 9.0)
    return st, hc, rc


def safe_mask(track, pos, yaw, v, w, humans_next, robots_next):
    """Devuelve (N, A) booleano: acción segura en el siguiente paso.
    Una acción es segura si respeta todos los márgenes, o si ya se está dentro
    de un margen y la acción no lo empeora (permite alejarse/girar sin bloqueo).
    Regla de desbloqueo: si ninguna acción cumple lo anterior (p. ej. dos robots
    ya en contacto), se aceptan las acciones que no reducen NINGUNA holgura."""
    N, A = len(pos), N_ACTIONS
    ok = np.ones((N, A), bool)
    relaxed = np.ones((N, A), bool)
    P = np.repeat(pos, A, 0); Y = np.repeat(yaw, A); V = np.repeat(v, A); W = np.repeat(w, A)
    av = np.tile(ACT_V, N); aw = np.tile(ACT_W, N)
    H = np.repeat(humans_next, A, 0); R = np.repeat(robots_next, A, 0)
    st0, hc0, rc0 = clearance(track, P, H, R)
    for frac in (0.5, 1.0):
        p, _ = predict(P, Y, V, W, av, aw, frac)
        st, hc, rc = clearance(track, p, H, R)
        nw_s, nw_h, nw_r = st >= st0 - 1e-4, hc >= hc0 - 1e-4, rc >= rc0 - 1e-4
        good = (((st >= MARGIN_STATIC) | (nw_s & (st > 0.03))) &
                ((hc >= MARGIN_HUMAN) | (nw_h & (hc > 0.05))) &
                ((rc >= MARGIN_ROBOT) | (nw_r & (rc > 0.03))))
        ok &= good.reshape(N, A)
        tol = 3e-3   # tolerancia al ruido de la marcha (paso en el lugar)
        relaxed &= (((st >= MARGIN_STATIC) | (st >= st0 - tol)) & ((hc >= MARGIN_HUMAN) | (hc >= hc0 - tol)) &
                    ((rc >= MARGIN_ROBOT) | (rc >= rc0 - tol))).reshape(N, A)
    stuck = ~ok.any(1)
    ok[stuck] = relaxed[stuck]
    return ok


def shield(track, actions, pos, yaw, v, w, humans_next, robots_next, prefs=None):
    """Filtra las acciones. prefs (N,A): preferencia de la política (probabilidades
    o valores Q). Devuelve (acciones_ejecutadas, intervino(bool N), mask)."""
    mask = safe_mask(track, pos, yaw, v, w, humans_next, robots_next)
    N = len(actions)
    proposed_ok = mask[np.arange(N), actions]
    out = actions.copy()
    bad = np.where(~proposed_ok)[0]
    for i in bad:
        m = mask[i]
        if not m.any():
            out[i] = STOP
            continue
        if prefs is not None:
            sc = np.where(m, prefs[i], -np.inf)
        else:   # acción segura "más parecida" a la propuesta
            d = np.abs(ACT_V - ACT_V[actions[i]]) / V_MAX + np.abs(ACT_W - ACT_W[actions[i]]) / W_MAX
            sc = np.where(m, -d, -np.inf)
        out[i] = int(np.argmax(sc))
    return out, ~proposed_ok, mask


def cost(st, hc, rc, collided, fell):
    """Señal de costo por paso (CMDP)."""
    c = (st < 0.03).astype(float) + (hc < 0.15).astype(float) + (rc < 0.04).astype(float)
    return c + collided.astype(float) * 2.0 + fell.astype(float)


class Lagrangian:
    """λ <- max(0, λ + lr (E[c] - d))  (ascenso dual)."""
    def __init__(self, limit=COST_LIMIT, lr=0.05, init=0.5, max_lam=3.0, enabled=True):
        self.limit, self.lr, self.lam, self.max_lam, self.enabled = limit, lr, init, max_lam, enabled
        if not enabled:
            self.lam = 0.0

    def update(self, mean_cost):
        if self.enabled:
            self.lam = float(np.clip(self.lam + self.lr * (mean_cost - self.limit), 0.0, self.max_lam))
        return self.lam
