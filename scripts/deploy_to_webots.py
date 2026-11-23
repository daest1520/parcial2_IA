"""Genera los mundos de Webots y copia las políticas entrenadas de cada
experimento a webots/policies/<mundo>/<ALGO>.npz (E01 también como política por defecto)."""
import os, sys, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_world import WORLDS, world_tracks, build_world, ROOT
from race_core.track import Track

pol = os.path.join(ROOT, "webots", "policies")
for w, cfg in world_tracks().items():
    e = WORLDS[w]
    src = os.path.join(ROOT, "results", "experiments", e)
    os.makedirs(os.path.join(pol, w), exist_ok=True)
    for a in ["DAC", "PPO", "DQN", "A2C"]:
        shutil.copy(os.path.join(src, f"{a}.npz"), os.path.join(pol, w, f"{a}.npz"))
        if e == "E01":
            shutil.copy(os.path.join(src, f"{a}.npz"), os.path.join(pol, f"{a}.npz"))
    print(build_world(Track(**cfg), w), "<-", e)
