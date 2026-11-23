"""
Definición paramétrica de la pista de carreras (compartida por el simulador de
entrenamiento y por los controladores de Webots).

La pista es un corredor tipo serpentina con 3 rectas, 2 curvas de 180° y una
chicana sinusoidal en la recta central. Incluye:
  * paredes laterales (segmentos),
  * obstáculos estáticos (cajas y cilindros),
  * un peatón (humano) que cruza el corredor,
  * zonas con condiciones distintas de piso (resbaladiza / rugosa),
  * línea de salida y meta configurables a lo largo de la pista.

Sólo depende de numpy para poder usarse dentro de Webots.
"""
import json
import math
import numpy as np

SIZES = {            # (Lx, Ly) en metros (escala de un humanoide de 0.45 m)
    "S": (5.0, 3.6),
    "M": (6.5, 4.4),
    "L": (8.0, 5.2),
}

ROBOT_RADIUS = 0.15      # radio de la huella del DARwIn-OP (m)
HUMAN_RADIUS = 0.25
WALL_THICK = 0.05


def _arc(cx, cy, r, a0, a1, n):
    t = np.linspace(a0, a1, n)
    return np.stack([cx + r * np.cos(t), cy + r * np.sin(t)], 1)


def _resample(pts, ds=0.05):
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    n = max(2, int(s[-1] / ds) + 1)
    si = np.linspace(0, s[-1], n)
    return np.stack([np.interp(si, s, pts[:, 0]), np.interp(si, s, pts[:, 1])], 1), si


class Track:
    def __init__(self, size="S", start_frac=0.0, goal_frac=1.0, reverse=False,
                 width=1.0, margin=0.6, chicane_amp=0.2, n_obstacles=None,
                 pedestrian=True, zones=True, seed=0):
        self.size = size
        self.Lx, self.Ly = SIZES[size]
        self.width = width
        self.cfg = dict(size=size, start_frac=start_frac, goal_frac=goal_frac,
                        reverse=reverse, width=width, margin=margin,
                        chicane_amp=chicane_amp, n_obstacles=n_obstacles,
                        pedestrian=pedestrian, zones=zones, seed=seed)
        rng = np.random.RandomState(seed)
        m = margin
        R = (self.Ly - 2 * m) / 4.0
        xa, xb = m + R, self.Lx - m - R
        y1, y2, y3 = m, m + 2 * R, m + 4 * R
        n_arc = 60
        # recta 1
        run1 = np.stack([np.linspace(0.05, xb, 40), np.full(40, y1)], 1)
        turn1 = _arc(xb, y1 + R, R, -math.pi / 2, math.pi / 2, n_arc)
        xs = np.linspace(xb, xa, 80)
        u = (xb - xs) / (xb - xa)
        run2 = np.stack([xs, y2 + chicane_amp * np.sin(2 * math.pi * u) * np.sin(math.pi * u)], 1)
        turn2 = _arc(xa, y2 + R, R, 3 * math.pi / 2, math.pi / 2, n_arc)
        run3 = np.stack([np.linspace(xa, self.Lx - 0.3, 40), np.full(40, y3)], 1)
        full = np.concatenate([run1, turn1[1:], run2[1:], turn2[1:], run3[1:]])
        full, s_full = _resample(full, 0.05)
        self.full_center = full
        self.full_length = s_full[-1]

        # ---------------- paredes (sobre toda la pista) ----------------
        self.walls = self._build_walls(full, width / 2)

        # ---------------- obstáculos ----------------
        L = self.full_length
        if n_obstacles is None:
            n_obstacles = int(round(L / 2.5))
        self.obstacles = []          # dict(kind, x, y, r | (hx, hy, yaw))
        fracs = np.linspace(0.12, 0.92, n_obstacles)
        fracs = fracs + rng.uniform(-0.02, 0.02, size=n_obstacles)
        lat_choices = [0.5, -0.5, 0.25, -0.5, 0.5]
        for i, f in enumerate(fracs):
            p, t, nrm = self._frame(full, s_full, f * L)
            lat = lat_choices[(i + seed) % len(lat_choices)] * (width / 2)
            c = p + nrm * lat
            small = abs(lat) < 0.2
            if (i + seed) % 2 == 0:
                hx = hy = rng.uniform(0.06, 0.08) if small else rng.uniform(0.09, 0.12)
                yaw = math.atan2(t[1], t[0])
                self.obstacles.append(dict(kind="box", x=float(c[0]), y=float(c[1]),
                                           hx=float(hx), hy=float(hy), yaw=float(yaw)))
            else:
                self.obstacles.append(dict(kind="cyl", x=float(c[0]), y=float(c[1]),
                                           r=float(rng.uniform(0.06, 0.08) if small else rng.uniform(0.09, 0.12))))

        # ---------------- peatón (humano) ----------------
        self.pedestrians = []
        if pedestrian:
            for f in ([0.47] if size == "S" else [0.47, 0.8]):
                p, t, nrm = self._frame(full, s_full, f * L)
                # cruza la pista de lado a lado y espera fuera del corredor
                a = p + nrm * (width / 2 + 0.3)
                b = p - nrm * (width / 2 + 0.3)
                self.pedestrians.append(dict(a=a.tolist(), b=b.tolist(), speed=0.25, pause=6.0))

        # ---------------- zonas de piso ----------------
        self.zones = []
        if zones:
            for kind, f0, f1 in [("slippery", 0.22, 0.30), ("rough", 0.58, 0.68)]:
                self.zones.append(dict(kind=kind, s0=f0 * L, s1=f1 * L))

        # ---------------- salida / meta ----------------
        center = full.copy()
        s_c = s_full.copy()
        if reverse:
            center = center[::-1].copy()
            s_c = (L - s_c)[::-1].copy()
            # las zonas se expresan en la longitud de la pista completa original
        i0 = int(np.searchsorted(s_c, start_frac * L))
        i1 = int(np.searchsorted(s_c, goal_frac * L)) - 1
        i1 = max(i1, i0 + 10)
        self.center = center[i0:i1 + 1]
        self.s = s_c[i0:i1 + 1] - s_c[i0]
        self.length = float(self.s[-1])
        self._reverse = reverse
        self._s_offset = s_c[i0]
        tang = np.gradient(self.center, axis=0)
        self.tangent = tang / (np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9)
        self.normal = np.stack([-self.tangent[:, 1], self.tangent[:, 0]], 1)
        self.start = self.center[0]
        self.start_yaw = float(math.atan2(self.tangent[0, 1], self.tangent[0, 0]))
        self.goal = self.center[-1]
        # zonas en coordenadas de este recorrido
        self.zone_ranges = []
        for z in self.zones:
            a, b = z["s0"], z["s1"]
            if reverse:
                a, b = L - b, L - a
            a, b = a - self._s_offset, b - self._s_offset
            self.zone_ranges.append((z["kind"], a, b))

        # los puestos de la parrilla de salida quedan libres de obstáculos
        slots = np.array([[x, y] for x, y, _ in self.grid_positions(4)])
        keep = []
        for o in self.obstacles:
            rad = o.get("r", math.hypot(o.get("hx", 0), o.get("hy", 0)))
            dmin = np.linalg.norm(slots - np.array([o["x"], o["y"]]), axis=1).min()
            if dmin > rad + ROBOT_RADIUS + 0.08:
                keep.append(o)
        self.obstacles = keep
        # segmentos estáticos totales (paredes + cajas) y círculos estáticos
        segs = [self.walls]
        circ = []
        for o in self.obstacles:
            if o["kind"] == "box":
                segs.append(_box_segments(o))
            else:
                circ.append([o["x"], o["y"], o["r"]])
        # paredes de cierre en la salida y en la meta (detrás de la salida)
        segs.append(self._cap(0, back=True))
        self.static_segs = np.concatenate(segs, 0).astype(np.float64)
        self.static_circles = np.array(circ, np.float64).reshape(-1, 3)
        self._build_sdf()
        self._build_free_lateral()

    def _build_free_lateral(self):
        """Desplazamiento lateral con mayor holgura a lo largo de la pista
        (línea de carrera usada por los robots fantasma)."""
        lats = np.linspace(-0.3, 0.3, 25)
        P = self.center[:, None, :] + self.normal[:, None, :] * lats[None, :, None]
        cl = self.sdf_query(P)
        # mirar 0.5 m hacia adelante para anticipar obstáculos
        n = len(self.center)
        best = np.zeros(n)
        for i in range(n):
            w = cl[i:min(n, i + 10)].min(0)
            best[i] = lats[np.argmax(w - 0.05 * np.abs(lats))]
        k = np.ones(15) / 15
        self.free_lat = np.convolve(np.pad(best, 7, mode="edge"), k, mode="valid")

    # ------------------------------------------------------------------
    def _build_sdf(self, res=0.02):
        """Campo de distancia a la geometría estática (consulta O(1))."""
        self.sdf_res = res
        nx, ny = int(self.Lx / res) + 3, int(self.Ly / res) + 3
        self.sdf_origin = np.array([-res, -res])
        xs = self.sdf_origin[0] + np.arange(nx) * res
        ys = self.sdf_origin[1] + np.arange(ny) * res
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        P = np.stack([X.ravel(), Y.ravel()], 1)
        d = np.full(len(P), 1e9)
        segs = self.static_segs
        for k0 in range(0, len(segs), 64):
            sg = segs[k0:k0 + 64]
            a = sg[None, :, :2]; b = sg[None, :, 2:]; ab = b - a
            for c0 in range(0, len(P), 40000):
                pp = P[c0:c0 + 40000, None, :]
                t = np.clip(((pp - a) * ab).sum(-1) / ((ab * ab).sum(-1) + 1e-12), 0, 1)
                q = a + ab * t[..., None]
                d[c0:c0 + 40000] = np.minimum(d[c0:c0 + 40000], np.sqrt(((pp - q) ** 2).sum(-1)).min(1))
        for cx, cy, r in self.static_circles:
            d = np.minimum(d, np.sqrt((P[:, 0] - cx) ** 2 + (P[:, 1] - cy) ** 2) - r)
        self.sdf = d.reshape(nx, ny)
        self._sdf_flat = np.ascontiguousarray(self.sdf).ravel()

    def sdf_query(self, p, nearest=False):
        """distancia desde puntos p (...,2) a la geometría estática (0 fuera del mapa)."""
        S = self.sdf
        nx, ny = S.shape
        gx = (p[..., 0] - self.sdf_origin[0]) * (1.0 / self.sdf_res)
        gy = (p[..., 1] - self.sdf_origin[1]) * (1.0 / self.sdf_res)
        out = (gx < 0) | (gy < 0) | (gx > nx - 1.001) | (gy > ny - 1.001)
        gx = np.minimum(np.maximum(gx, 0), nx - 1.001); gy = np.minimum(np.maximum(gy, 0), ny - 1.001)
        flat = self._sdf_flat
        if nearest:
            v = flat[(gx + 0.5).astype(np.intp) * ny + (gy + 0.5).astype(np.intp)]
        else:
            ix = gx.astype(np.intp); iy = gy.astype(np.intp)
            fx = gx - ix; fy = gy - iy
            k = ix * ny + iy
            v = ((flat[k] * (1 - fx) + flat[k + ny] * fx) * (1 - fy) +
                 (flat[k + 1] * (1 - fx) + flat[k + ny + 1] * fx) * fy)
        v[out] = 0.0
        return v

    # ------------------------------------------------------------------
    def _frame(self, pts, s, at):
        i = int(np.clip(np.searchsorted(s, at), 1, len(pts) - 2))
        p = pts[i]
        t = pts[i + 1] - pts[i - 1]
        t = t / np.linalg.norm(t)
        return p, t, np.array([-t[1], t[0]])

    def _build_walls(self, pts, h):
        t = np.gradient(pts, axis=0)
        t /= np.linalg.norm(t, axis=1, keepdims=True)
        n = np.stack([-t[:, 1], t[:, 0]], 1)
        left = pts + n * h
        right = pts - n * h
        step = 4    # 20 cm por segmento
        idx = list(range(0, len(pts), step))
        if idx[-1] != len(pts) - 1:
            idx.append(len(pts) - 1)
        L = np.array([[left[a, 0], left[a, 1], left[b, 0], left[b, 1]] for a, b in zip(idx[:-1], idx[1:])])
        Rr = np.array([[right[a, 0], right[a, 1], right[b, 0], right[b, 1]] for a, b in zip(idx[:-1], idx[1:])])
        self.wall_left, self.wall_right = left[idx], right[idx]
        return np.concatenate([L, Rr], 0)

    def _cap(self, i, back=True):
        """Pared transversal 0.6 m detrás de la línea de salida."""
        p, t, n = self.center[i], self.tangent[i], self.normal[i]
        c = p - t * 0.05
        h = self.width / 2
        a, b = c + n * h, c - n * h
        return np.array([[a[0], a[1], b[0], b[1]]])

    # ------------------------------------------------------------------
    def grid_positions(self, n=4):
        """Parrilla de salida: 2 carriles x 2 filas (escalonada). Las filas se
        ubican SOBRE el arco de la pista (válido aunque la salida esté en curva)."""
        out = []
        lat = [0.22, -0.22, 0.22, -0.22]
        row_idx = [14, 14, 6, 6]   # fila delantera a 0.70 m y trasera a 0.30 m de la línea
        for k in range(n):
            i = row_idx[k % 4]
            p, nr = self.center[i], self.normal[i]
            q = p + nr * lat[k % 4]
            yaw = float(math.atan2(self.tangent[i, 1], self.tangent[i, 0]))
            out.append((float(q[0]), float(q[1]), yaw))
        return out

    def max_steps(self, dt, vmax):
        return int(3.5 * self.length / (vmax * dt)) + 40

    def to_json(self, path=None):
        d = dict(cfg=self.cfg, Lx=self.Lx, Ly=self.Ly, width=self.width,
                 length=self.length, start=self.start.tolist(), start_yaw=self.start_yaw,
                 goal=self.goal.tolist(), obstacles=self.obstacles,
                 pedestrians=self.pedestrians, zones=self.zone_ranges,
                 wall_left=self.wall_left.tolist(), wall_right=self.wall_right.tolist(),
                 grid=self.grid_positions())
        if path:
            with open(path, "w") as f:
                json.dump(d, f)
        return d

    @staticmethod
    def from_json(path):
        with open(path) as f:
            d = json.load(f)
        return Track(**d["cfg"])


def _box_segments(o):
    c, s = math.cos(o["yaw"]), math.sin(o["yaw"])
    hx, hy = o["hx"], o["hy"]
    loc = np.array([[hx, hy], [-hx, hy], [-hx, -hy], [hx, -hy]])
    R = np.array([[c, -s], [s, c]])
    P = loc @ R.T + np.array([o["x"], o["y"]])
    return np.array([[P[k, 0], P[k, 1], P[(k + 1) % 4, 0], P[(k + 1) % 4, 1]] for k in range(4)])


# ------------------------------------------------------------------------
#  Proyección sobre la línea central (progreso a lo largo de la pista)
# ------------------------------------------------------------------------
def project(track, xy, prev_idx=None, window=60):
    """xy: (N,2). Devuelve índice del punto más cercano de la línea central,
    progreso s (m) y desplazamiento lateral con signo (m)."""
    C = track.center
    if prev_idx is None:
        d = ((xy[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        idx = d.argmin(1)
    else:
        offs = np.arange(-window, window + 1)
        cand = np.clip(prev_idx[:, None] + offs[None, :], 0, len(C) - 1)
        d = ((xy[:, None, :] - C[cand]) ** 2).sum(-1)
        idx = cand[np.arange(len(xy)), d.argmin(1)]
    lat = ((xy - C[idx]) * track.normal[idx]).sum(1)
    return idx, track.s[idx], lat
