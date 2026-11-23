"""
Agrega los resultados de results/experiments/* y genera tablas (CSV) y figuras
(PNG) para el informe: results/tables/*.csv, results/figures/*.png
"""
import os, sys, json, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from race_core.track import Track, _box_segments

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(ROOT, "results", "experiments")
TAB = os.path.join(ROOT, "results", "tables"); FIG = os.path.join(ROOT, "results", "figures")
os.makedirs(TAB, exist_ok=True); os.makedirs(FIG, exist_ok=True)
ALG = ["DAC", "PPO", "DQN", "A2C"]
COL = {"DAC": "#2a78d6", "PPO": "#eb6834", "DQN": "#1baf7a", "A2C": "#eda100"}   # paleta validada (CVD)
MARK = {"DAC": "o", "PPO": "s", "DQN": "^", "A2C": "D"}
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
                     "axes.edgecolor": "#888", "legend.frameon": False, "figure.dpi": 200,
                     "lines.linewidth": 1.6})


def load():
    S = {}
    for d in sorted(glob.glob(os.path.join(EXP, "E*"))):
        f = os.path.join(d, "summary.json")
        if os.path.exists(f):
            with open(f) as fh:
                S[os.path.basename(d)] = json.load(fh)
            rf = os.path.join(d, "races_final.json")
            if os.path.exists(rf):   # carreras con la lógica de despliegue (Webots)
                rr = json.load(open(rf))
                S[os.path.basename(d)]["races"] = rr["races"]
                S[os.path.basename(d)]["race0_traj"] = rr["traj0"]
    return S


def per_algo_table(S):
    rows = []
    for e, s in S.items():
        for a in ALG:
            ev = s["algos"][a]["eval"]
            succ = [r["success"] for r in ev]
            t = [r["time"] for r in ev if r["success"]]
            rows.append(dict(exp=e, factor=s["factor"], desc=s["desc"], algo=a,
                             success=np.mean(succ), ret=np.mean([r["ret"] for r in ev]),
                             ret_std=np.std([r["ret"] for r in ev]),
                             time=np.mean(t) if t else np.nan, progress=np.mean([r["progress"] for r in ev]),
                             collisions=np.mean([r["collisions"] for r in ev]), falls=np.mean([r["falls"] for r in ev]),
                             shield=np.mean([r["shield"] for r in ev]), cost=np.mean([r["cost"] for r in ev]),
                             train_time=s["algos"][a]["train_time"], lam=s["algos"][a]["lam"]))
    return pd.DataFrame(rows)


def race_table(S):
    rows = []
    for e, s in S.items():
        for r in s["races"]:
            for pos, a in enumerate(ALG):
                place = r["order"].index(a) + 1 if a in r["order"] else np.nan
                rows.append(dict(exp=e, algo=a, win=int(r["winner"] == a), place=place,
                                 finished=int(a in r["order"]), time=r["finish_time"][a],
                                 progress=r["progress"][a], collisions=r["collisions"][a], falls=r["falls"][a]))
    return pd.DataFrame(rows)


def learning_curves(S, exps=None, fname="curvas_aprendizaje.png", title=None):
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.4))
    grid = np.linspace(0, 400_000, 41)
    for a in ALG:
        Rs, Ss = [], []
        for e, s in S.items():
            if exps and e not in exps:
                continue
            f = os.path.join(EXP, e, f"{a}_train_episodes.json")
            ep = np.array(json.load(open(f)))
            if len(ep) < 5:
                continue
            st, ret, suc = ep[:, 0], ep[:, 1], ep[:, 2]
            k = min(25, len(ep))
            rs = np.convolve(ret, np.ones(k) / k, "valid"); ss = np.convolve(suc, np.ones(k) / k, "valid")
            sv = st[k - 1:]                       # media móvil causal (sin artefacto de borde)
            Rs.append(np.interp(grid, sv, rs)); Ss.append(np.interp(grid, sv, ss))
        if not Rs:
            continue
        for ax, Y in ((axs[0], np.array(Rs)), (axs[1], np.array(Ss) * 100)):
            m = Y.mean(0); se = Y.std(0) / np.sqrt(len(Y))
            ax.plot(grid / 1e3, m, color=COL[a], label=a)
            ax.fill_between(grid / 1e3, m - se, m + se, color=COL[a], alpha=0.15, lw=0)
    axs[0].set(xlabel="pasos de entrenamiento (miles)", ylabel="retorno por episodio")
    axs[1].set(xlabel="pasos de entrenamiento (miles)", ylabel="tasa de éxito (%)", ylim=(0, 100))
    axs[1].legend(ncol=4, loc="upper left", fontsize=7)
    if title:
        fig.suptitle(title, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, fname)); plt.close(fig)


def bars_by_exp(df, metric, ylabel, fname, scale=1.0, ylim=None):
    exps = sorted(df.exp.unique())
    x = np.arange(len(exps)); w = 0.2
    fig, ax = plt.subplots(figsize=(7.0, 2.3))
    for i, a in enumerate(ALG):
        v = [df[(df.exp == e) & (df.algo == a)][metric].values[0] * scale for e in exps]
        ax.bar(x + (i - 1.5) * w, v, w * 0.9, color=COL[a], label=a, edgecolor="white", linewidth=0.4)
    ax.set_xticks(x); ax.set_xticklabels(exps, rotation=0, fontsize=6.5)
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.18), fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, fname)); plt.close(fig)


def factor_summary(df):
    g = df.groupby(["factor", "algo"]).agg(success=("success", "mean"), ret=("ret", "mean"),
                                          collisions=("collisions", "mean")).reset_index()
    order = ["base", "tamaño", "salida", "meta", "parámetros", "safe learning"]
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    x = np.arange(len(order)); w = 0.2
    for i, a in enumerate(ALG):
        v = [g[(g.factor == f) & (g.algo == a)].success.values[0] * 100 if len(g[(g.factor == f) & (g.algo == a)]) else 0 for f in order]
        ax.bar(x + (i - 1.5) * w, v, w * 0.9, color=COL[a], label=a, edgecolor="white", linewidth=0.4)
    ax.set_xticks(x); ax.set_xticklabels(["base", "tamaño", "salida", "meta", "parám.", "seguridad"], fontsize=6.5)
    ax.set_ylabel("éxito medio (%)"); ax.set_ylim(0, 100)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.2), fontsize=6.5)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "exito_por_factor.png")); plt.close(fig)
    return g


def boxplot_returns(S):
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    data = []
    for a in ALG:
        data.append([r["ret"] for e, s in S.items() for r in s["algos"][a]["eval"]])
    bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=False,
                    medianprops=dict(color="#222", linewidth=1.2))
    for patch, a in zip(bp["boxes"], ALG):
        patch.set_facecolor(COL[a]); patch.set_alpha(0.8); patch.set_edgecolor("#444")
    ax.set_xticks(range(1, 5)); ax.set_xticklabels(ALG)
    ax.set_ylabel("recompensa acumulada")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "boxplot_retorno.png")); plt.close(fig)


def race_plot(rt):
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.2))
    wins = rt.groupby("algo").win.sum().reindex(ALG)
    n = rt.groupby("exp").ngroups * 0 + len(rt) / 4
    axs[0].bar(ALG, wins.values / n * 100, color=[COL[a] for a in ALG], width=0.6)
    for i, v in enumerate(wins.values):
        axs[0].text(i, v / n * 100 + 1, f"{int(v)}", ha="center", fontsize=7, color="#333")
    axs[0].set_ylabel("carreras ganadas (%)")
    pl = rt.copy(); pl["place"] = pl["place"].fillna(5)
    for i, a in enumerate(ALG):
        c = pl[pl.algo == a].place.value_counts().reindex([1, 2, 3, 4, 5]).fillna(0) / (len(pl) / 4) * 100
        axs[1].plot(["1°", "2°", "3°", "4°", "no llegó"], c.values, marker=MARK[a], color=COL[a], label=a, ms=5)
    axs[1].set_ylabel("% de carreras"); axs[1].legend(ncol=2, fontsize=6.5)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "carreras.png")); plt.close(fig)


def draw_track(ax, tr):
    ax.plot(tr.wall_left[:, 0], tr.wall_left[:, 1], color="#333", lw=1.2)
    ax.plot(tr.wall_right[:, 0], tr.wall_right[:, 1], color="#333", lw=1.2)
    for kind, a, b in tr.zone_ranges:
        m = (tr.s >= a) & (tr.s <= b)
        ax.plot(tr.center[m, 0], tr.center[m, 1], lw=11, alpha=0.25, solid_capstyle="butt",
                color="#7fb2e5" if kind == "slippery" else "#a0785a")
    for o in tr.obstacles:
        if o["kind"] == "cyl":
            ax.add_patch(plt.Circle((o["x"], o["y"]), o["r"], color="#555"))
        else:
            P = _box_segments(o); ax.fill(P[:, 0], P[:, 1], color="#555")
    for p in tr.pedestrians:
        a_, b_ = np.array(p["a"]), np.array(p["b"])
        ax.plot([a_[0], b_[0]], [a_[1], b_[1]], ls=":", color="#b0336e", lw=1.5)
    ax.plot(*tr.start, marker=">", color="#2e7d32", ms=7); ax.plot(*tr.goal, marker="*", color="#333", ms=10)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)


def race_trajectory(S, e="E01"):
    s = S[e]; tr = Track(**s["track"])
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    draw_track(ax, tr)
    P = np.array([t["pos"] for t in s["race0_traj"]])
    for j, a in enumerate(ALG):
        ax.plot(P[:, j, 0], P[:, j, 1], color=COL[a], lw=1.1, label=a)
    ax.legend(ncol=4, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout(); fig.savefig(os.path.join(FIG, f"trayectoria_carrera_{e}.png")); plt.close(fig)


def tracks_fig():
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 1.9), gridspec_kw=dict(width_ratios=[5, 6.5, 8]))
    for ax, sz in zip(axs, "SML"):
        tr = Track(size=sz); draw_track(ax, tr)
        ax.set_title(f"{sz}: {tr.Lx}×{tr.Ly} m, L={tr.length:.1f} m", fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "pistas.png")); plt.close(fig)


def dac_diag(S, e="E01"):
    d = S[e]["algos"]["DAC"]["diag"]
    if not d:
        return
    st = np.array([x["step"] for x in d]) / 1e3
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 1.8))
    for ax, k, lab in zip(axs, ["critic_std", "ask_rate", "lam"], ["desacuerdo σ_V", "tasa de consulta", "λ (Lagrangiano)"]):
        ax.plot(st, [x[k] for x in d], color=COL["DAC"], lw=1.2); ax.set_title(lab, fontsize=7)
        ax.set_xlabel("pasos (miles)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "diagnostico_dac.png")); plt.close(fig)


if __name__ == "__main__":
    S = load()
    print("experimentos:", list(S))
    df = per_algo_table(S); df.to_csv(os.path.join(TAB, "evaluacion_por_experimento.csv"), index=False)
    rt = race_table(S); rt.to_csv(os.path.join(TAB, "carreras.csv"), index=False)
    agg = df.groupby("algo").agg(success=("success", "mean"), ret=("ret", "mean"), time=("time", "mean"),
                                 progress=("progress", "mean"), collisions=("collisions", "mean"),
                                 falls=("falls", "mean"), shield=("shield", "mean"), cost=("cost", "mean"),
                                 train_time=("train_time", "mean")).reindex(ALG)
    ragg = rt.groupby("algo").agg(wins=("win", "sum"), finish=("finished", "mean"), place=("place", "mean"),
                                  rcol=("collisions", "mean")).reindex(ALG)
    agg = agg.join(ragg); agg.to_csv(os.path.join(TAB, "resumen_global.csv"))
    print(agg.round(3).to_string())
    g = factor_summary(df); g.to_csv(os.path.join(TAB, "por_factor.csv"), index=False)
    print(g.pivot(index="factor", columns="algo", values="success").round(2))
    learning_curves(S, [e for e in S if S[e]["factor"] not in ("safe learning",)])
    bars_by_exp(df, "success", "tasa de éxito (%)", "exito_por_experimento.png", 100, (0, 100))
    bars_by_exp(df, "ret", "recompensa acumulada", "retorno_por_experimento.png")
    boxplot_returns(S)
    race_plot(rt)
    tracks_fig()
    if "E01" in S:
        race_trajectory(S, "E01"); dac_diag(S, "E01")
    piv = df.pivot(index="exp", columns="algo", values="success")[ALG]
    print(piv.round(2).to_string())
