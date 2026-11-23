"""
Genera el mundo de Webots (.wbt) y la descripción de la pista (.json) a partir
de la misma clase Track usada para entrenar. Así el entorno de Webots y el
gemelo de entrenamiento son geométricamente idénticos.

Uso:
  python scripts/make_world.py --size M --start 0 --goal 1 --name race_M
  python scripts/make_world.py --all          # genera los mundos de los experimentos
"""
import os, sys, math, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from race_core.track import Track, _box_segments

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WB = os.path.join(ROOT, "webots")
R = "https://raw.githubusercontent.com/cyberbotics/webots/R2025a/projects"

AGENTS = [("DAC", "0.85 0.1 0.1"), ("PPO", "0.1 0.35 0.9"), ("DQN", "0.1 0.7 0.2"), ("A2C", "0.95 0.65 0.05")]


def f(x):
    return f"{x:.4f}"


def wall_solid(pts, name, h=0.3, th=0.05):
    kids, bos = [], []
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        L = math.hypot(x1 - x0, y1 - y0)
        if L < 1e-4:
            continue
        yaw = math.atan2(y1 - y0, x1 - x0)
        tr = f"{f((x0 + x1) / 2)} {f((y0 + y1) / 2)} {f(h / 2)}"
        rot = f"0 0 1 {f(yaw)}"
        kids.append(f"Pose {{ translation {tr} rotation {rot} children [ Shape {{ appearance USE WALL_APP geometry Box {{ size {f(L + 0.02)} {th} {h} }} }} ] }}")
        bos.append(f"Pose {{ translation {tr} rotation {rot} children [ Box {{ size {f(L + 0.02)} {th} {h} }} ] }}")
    return (f"Solid {{ name \"{name}\" children [\n    " + "\n    ".join(kids) +
            f"\n  ] boundingObject Group {{ children [\n    " + "\n    ".join(bos) + "\n  ] } }\n")


def strip_plates(tr, s0, s1, app_def, name, contact=None, z=0.0, h=0.004, width=None, bounding=True):
    width = width or tr.width
    kids, bos = [], []
    import numpy as np
    m = (tr.s >= s0) & (tr.s <= s1)
    idx = np.where(m)[0]
    for a, b in zip(idx[:-1:2], idx[2::2]):
        p0, p1 = tr.center[a], tr.center[b]
        L = float(np.linalg.norm(p1 - p0)); yaw = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
        t = f"{f((p0[0] + p1[0]) / 2)} {f((p0[1] + p1[1]) / 2)} {f(z)}"
        kids.append(f"Pose {{ translation {t} rotation 0 0 1 {f(yaw)} children [ Shape {{ appearance USE {app_def} geometry Box {{ size {f(L + 0.03)} {f(width)} {h} }} }} ] }}")
        bos.append(f"Pose {{ translation {t} rotation 0 0 1 {f(yaw)} children [ Box {{ size {f(L + 0.03)} {f(width)} {h} }} ] }}")
    if not kids:
        return ""
    # visual: malla continua (sin "abanico" en las curvas); física: cajas
    ii = idx[::2]
    L_ = tr.center[ii] + tr.normal[ii] * width / 2; R_ = tr.center[ii] - tr.normal[ii] * width / 2
    zt = z + h / 2 + 0.0005
    pts = " ".join(f"{f(q[0])} {f(q[1])} {zt:.4f}" for q in np.concatenate([L_, R_]))
    n = len(ii)
    faces = " ".join(f"{k} {n + k} {n + k + 1} {k + 1} -1" for k in range(n - 1))
    kids = [f"Shape {{ appearance USE {app_def} geometry IndexedFaceSet {{ coord Coordinate {{ point [ {pts} ] }} coordIndex [ {faces} ] }} }}"]
    s = f"Solid {{ name \"{name}\" children [\n    " + "\n    ".join(kids) + "\n  ]"
    if contact:
        s += f" contactMaterial \"{contact}\""
    if bounding:
        s += " boundingObject Group { children [\n    " + "\n    ".join(bos) + "\n  ] }"
    return s + " }\n"


def line_marker(tr, i, name, checker=False):
    p, t, n = tr.center[i], tr.tangent[i], tr.normal[i]
    yaw = math.atan2(t[1], t[0])
    kids = []
    if checker:
        k = 8
        for r in range(2):
            for c in range(k):
                off = -tr.width / 2 + (c + 0.5) * tr.width / k
                q = p + n * off + t * (r - 0.5) * 0.06
                col = "USE BLACK_APP" if (r + c) % 2 else "USE WHITE_APP"
                kids.append(f"Pose {{ translation {f(q[0])} {f(q[1])} 0.003 rotation 0 0 1 {f(yaw)} children [ Shape {{ appearance {col} geometry Box {{ size 0.06 {f(tr.width / k)} 0.002 }} }} ] }}")
    else:
        kids.append(f"Pose {{ translation {f(p[0])} {f(p[1])} 0.003 rotation 0 0 1 {f(yaw)} children [ Shape {{ appearance USE WHITE_APP geometry Box {{ size 0.05 {f(tr.width)} 0.002 }} }} ] }}")
    return f"Solid {{ name \"{name}\" children [ " + " ".join(kids) + " ] }\n"


def build_world(tr: Track, name, title=None):
    gx = tr.grid_positions(4)
    out = [f"""#VRML_SIM R2025a utf8
EXTERNPROTO "{R}/objects/backgrounds/protos/TexturedBackground.proto"
EXTERNPROTO "{R}/objects/backgrounds/protos/TexturedBackgroundLight.proto"
EXTERNPROTO "{R}/objects/floors/protos/Floor.proto"
EXTERNPROTO "{R}/robots/robotis/darwin-op/protos/Darwin-op.proto"
EXTERNPROTO "{R}/humans/pedestrian/protos/Pedestrian.proto"

WorldInfo {{
  info [ "Parcial 2 IA - Carrera de DARwIn-OP: DAC (propio) vs PPO vs DQN vs A2C"
         "Pista {tr.size} {tr.Lx}x{tr.Ly} m, longitud {tr.length:.2f} m" ]
  title "{title or name}"
  basicTimeStep 16
  contactProperties [
    ContactProperties {{ material1 "slippery" coulombFriction [ 0.25 ] }}
    ContactProperties {{ material1 "rough" coulombFriction [ 1.2 ] softCFM 0.003 }}
  ]
}}
DEF VIEW Viewpoint {{
  orientation -0.3989 0.3989 0.8257 1.7611
  position {f(tr.Lx / 2)} {f(-tr.Ly * 0.45)} {f(max(tr.Lx, tr.Ly) * 0.85)}
}}
TexturedBackground {{ }}
TexturedBackgroundLight {{ }}
Floor {{
  translation {f(tr.Lx / 2)} {f(tr.Ly / 2)} 0
  size {f(tr.Lx + 2)} {f(tr.Ly + 2)}
  tileSize 0.5 0.5
}}
DEF APPEARANCES Group {{ children [
  Shape {{ appearance DEF WALL_APP PBRAppearance {{ baseColor 0.85 0.85 0.9 roughness 0.6 metalness 0 }} geometry Box {{ size 0.001 0.001 0.001 }} }}
  Shape {{ appearance DEF SLIP_APP PBRAppearance {{ baseColor 0.55 0.8 1 roughness 0.05 metalness 0 transparency 0.2 }} geometry Box {{ size 0.001 0.001 0.001 }} }}
  Shape {{ appearance DEF ROUGH_APP PBRAppearance {{ baseColor 0.45 0.32 0.2 roughness 1 metalness 0 }} geometry Box {{ size 0.001 0.001 0.001 }} }}
  Shape {{ appearance DEF WHITE_APP PBRAppearance {{ baseColor 1 1 1 roughness 1 metalness 0 }} geometry Box {{ size 0.001 0.001 0.001 }} }}
  Shape {{ appearance DEF BLACK_APP PBRAppearance {{ baseColor 0.05 0.05 0.05 roughness 1 metalness 0 }} geometry Box {{ size 0.001 0.001 0.001 }} }}
  Shape {{ appearance DEF OBST_APP PBRAppearance {{ baseColor 0.95 0.45 0.1 roughness 0.5 metalness 0 }} geometry Box {{ size 0.001 0.001 0.001 }} }}
  Shape {{ appearance DEF CYL_APP PBRAppearance {{ baseColor 0.85 0.1 0.15 roughness 0.5 metalness 0 }} geometry Box {{ size 0.001 0.001 0.001 }} }}
] }}
"""]
    out.append(wall_solid(tr.wall_left.tolist(), "pared_izquierda"))
    out.append(wall_solid(tr.wall_right.tolist(), "pared_derecha"))
    # pared de cierre detrás de la salida
    cap = tr.static_segs[-1]
    out.append(wall_solid([[cap[0], cap[1]], [cap[2], cap[3]]], "pared_salida"))
    for k, o in enumerate(tr.obstacles):
        if o["kind"] == "box":
            out.append(f"DEF OBST_{k} Solid {{ translation {f(o['x'])} {f(o['y'])} 0.15 rotation 0 0 1 {f(o['yaw'])} "
                       f"children [ Shape {{ appearance USE OBST_APP geometry DEF B{k} Box {{ size {f(2 * o['hx'])} {f(2 * o['hy'])} 0.3 }} }} ] "
                       f"name \"obstaculo_caja_{k}\" boundingObject USE B{k} }}\n")
        else:
            out.append(f"DEF OBST_{k} Solid {{ translation {f(o['x'])} {f(o['y'])} 0.2 "
                       f"children [ Shape {{ appearance USE CYL_APP geometry DEF C{k} Cylinder {{ radius {f(o['r'])} height 0.4 }} }} ] "
                       f"name \"obstaculo_cilindro_{k}\" boundingObject USE C{k} }}\n")
    for kind, a, b in tr.zone_ranges:
        app = "SLIP_APP" if kind == "slippery" else "ROUGH_APP"
        out.append(strip_plates(tr, max(a, 0), min(b, tr.length), app,
                                "zona_resbaladiza" if kind == "slippery" else "zona_rugosa", contact=kind))
    out.append(line_marker(tr, 0, "linea_salida"))
    out.append(line_marker(tr, len(tr.center) - 1, "meta", checker=True))
    for k, p in enumerate(tr.pedestrians):
        a, b = p["a"], p["b"]
        out.append(f"DEF HUMAN_{k} Pedestrian {{ translation {f(a[0])} {f(a[1])} 1.27 name \"peaton_{k}\" "
                   f"controller \"race_pedestrian\" controllerArgs [ \"--a={f(a[0])} {f(a[1])}\" \"--b={f(b[0])} {f(b[1])}\" "
                   f"\"--speed={p['speed']}\" \"--pause={p.get('pause', 6.0)}\" \"--phase={3.0 * k}\" ] }}\n")
    for k, (nm, col) in enumerate(AGENTS):
        x, y, yaw = gx[k]
        out.append(f"""DEF RACER_{nm} Darwin-op {{
  translation {f(x)} {f(y)} 0.236
  rotation 0 0 1 {f(yaw)}
  name "{nm}"
  controller "darwin_walker"
  channel {k + 1}
  plasticColor {col}
  bodySlot [ Pose {{ translation -0.06 0 0.02 children [ Shape {{ appearance PBRAppearance {{ baseColor {col} roughness 0.4 metalness 0 emissiveColor {col} }} geometry Box {{ size 0.03 0.12 0.08 }} }} ] }} ]
}}
""")
    out.append(f"""DEF RACE_BRAIN Robot {{
  name "race_brain"
  controller "race_brain"
  controllerArgs [ "--track=../../tracks/{name}.json" "--policies=../../policies" ]
  supervisor TRUE
  children [
    Emitter {{ name "emitter" channel 1 range -1 }}
    Receiver {{ name "receiver" channel -1 }}
  ]
}}
""")
    os.makedirs(os.path.join(WB, "worlds"), exist_ok=True)
    os.makedirs(os.path.join(WB, "tracks"), exist_ok=True)
    with open(os.path.join(WB, "worlds", f"{name}.wbt"), "w") as fh:
        fh.write("".join(out))
    tr.to_json(os.path.join(WB, "tracks", f"{name}.json"))
    return os.path.join(WB, "worlds", f"{name}.wbt")


WORLDS = {   # un mundo por cada experimento que cambia la pista (tamaño / salida / meta)
    "carrera_S": "E01", "carrera_M": "E02", "carrera_L": "E03",
    "carrera_S_salida15": "E04", "carrera_S_salida30": "E05", "carrera_S_inversa": "E06",
    "carrera_M_inversa": "E07", "carrera_S_meta60": "E08", "carrera_S_meta80": "E09",
    "carrera_M_s20_m80": "E10", "carrera_L_salida30": "E11",
}


def world_tracks():
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from run_experiments import EXPERIMENTS
    return {w: EXPERIMENTS[e]["track"] for w, e in WORLDS.items()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="M"); ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--goal", type=float, default=1.0); ap.add_argument("--reverse", action="store_true")
    ap.add_argument("--name", default=None); ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    if a.all:
        for nm, cfg in world_tracks().items():
            print(build_world(Track(**cfg), nm))
    else:
        nm = a.name or f"carrera_{a.size}"
        print(build_world(Track(size=a.size, start_frac=a.start, goal_frac=a.goal, reverse=a.reverse), nm))
