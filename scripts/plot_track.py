import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from race_core.track import Track
def draw(ax, tr, title=""):
    ax.plot(tr.wall_left[:,0], tr.wall_left[:,1], 'k-', lw=1.5); ax.plot(tr.wall_right[:,0], tr.wall_right[:,1], 'k-', lw=1.5)
    ax.plot(tr.center[:,0], tr.center[:,1], '--', color='gray', lw=.8)
    for kind,a,b in tr.zone_ranges:
        m=(tr.s>=a)&(tr.s<=b); ax.plot(tr.center[m,0], tr.center[m,1], lw=14, alpha=.25, color='tab:blue' if kind=='slippery' else 'tab:brown', solid_capstyle='butt')
    for o in tr.obstacles:
        if o['kind']=='cyl': ax.add_patch(plt.Circle((o['x'],o['y']),o['r'],color='tab:red'))
        else:
            from race_core.track import _box_segments
            P=_box_segments(o); ax.fill(P[:,0],P[:,1],color='tab:orange')
    for p in tr.pedestrians:
        a,b=np.array(p['a']),np.array(p['b']); ax.plot([a[0],b[0]],[a[1],b[1]],'m:',lw=2); ax.plot(*((a+b)/2),'m^',ms=9)
    for x,y,_ in tr.grid_positions(): ax.plot(x,y,'o',color='tab:green',ms=5)
    ax.plot(*tr.start,'gs',ms=8); ax.plot(*tr.goal,'r*',ms=14)
    ax.set_aspect('equal'); ax.set_title(title, fontsize=9); ax.set_xlim(0,tr.Lx); ax.set_ylim(0,tr.Ly); ax.grid(alpha=.2)
if __name__=="__main__":
    fig,axs=plt.subplots(1,3,figsize=(15,4))
    for ax,(sz) in zip(axs,"SML"):
        tr=Track(sz); draw(ax,tr,f"Pista {sz}: {tr.Lx}x{tr.Ly} m, L={tr.length:.1f} m, {len(tr.obstacles)} obst.")
    plt.tight_layout(); plt.savefig(sys.argv[1] if len(sys.argv)>1 else "tracks.png", dpi=130)
