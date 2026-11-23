import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
fig, ax = plt.subplots(figsize=(7.0, 2.9)); ax.set_xlim(0, 100); ax.set_ylim(0, 44); ax.axis("off")
def box(x, y, w, h, title, lines, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec="#555", lw=0.8))
    ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=7.2, weight="bold", color="#222")
    for i, l in enumerate(lines):
        ax.text(x + w / 2, y + h - 6.0 - i * 3.0, l, ha="center", va="top", fontsize=6.0, color="#333")
def arrow(x0, y0, x1, y1, txt="", dy=1.2):
    ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color="#444", lw=0.9))
    if txt: ax.text((x0 + x1) / 2, (y0 + y1) / 2 + dy, txt, ha="center", fontsize=5.8, color="#444")
box(1, 23, 27, 18, "Gemelo de entrenamiento", ["race_core/sim.py (numpy)", "marcha calibrada en Webots", "pista idéntica (Track)", "16 entornos en paralelo"], "#eef3fb")
box(1, 1, 27, 18, "Algoritmos (PyTorch)", ["DAC (Taller 2) · PPO", "DQN · A2C", "escudo + Lagrangiano", "20 experimentos"], "#fdf1e9")
box(36, 12, 26, 16, "Políticas .npz", ["MLP exportado", "inferencia sólo numpy", "(error < 1e-6 vs PyTorch)"], "#eef8f2")
box(70, 23, 29, 18, "race_brain (supervisor)", ["poses → observación 27-D", "π(a|s) de los 4 agentes", "escudo de seguridad", "meta, colisiones, caídas, log"], "#fdf6e3")
box(70, 1, 29, 16, "darwin_walker × 4 (C++)", ["RobotisOp2GaitManager", "rampa + watchdog", "levantarse tras caída"], "#f3eef9")
arrow(14.5, 22.5, 14.5, 19.6, "")
ax.text(16, 20.4, "experiencia / política", fontsize=5.8, color="#444")
arrow(28.5, 10, 35.5, 17, "exportar", dy=2.0)
arrow(62.5, 22, 69.5, 29, "cargar", dy=2.0)
arrow(84.5, 22.5, 84.5, 17.6)
ax.text(86, 19.6, "CMD x a (Emitter)", fontsize=5.8, color="#444")
ax.text(49, 39, "Webots R2025a: pista, obstáculos,\npeatón, zonas de piso, 4 DARwIn-OP", ha="center", va="center", fontsize=6.3, color="#444", style="italic")
arrow(62.5, 38, 69.5, 36)
fig.tight_layout(pad=0.2); fig.savefig("results/figures/arquitectura.png", dpi=220)
