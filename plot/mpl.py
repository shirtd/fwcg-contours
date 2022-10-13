import matplotlib.pyplot as plt
import numpy.linalg as la
import numpy as np
import util.util
# plt.ion()

def plot_poly(ax, P, T, visible=True, **kwargs):
    tp = {t : plt.Polygon(P[t,:], **kwargs) for t in T}
    util.util.lmap(lambda t: ax.add_patch(t), tp.values())
    if not visible:
        for t,p in tp.items():
            p.set_visible(False)
    return tp

def plot_edges(ax, P, E, visible=True, **kwargs):
    ep = {e : ax.plot(P[e,0], P[e,1], **kwargs)[0] for e in E}
    if not visible:
        for e,p in ep.items():
            p.set_visible(False)
    return ep

def plot_balls(ax, P, F, **kwargs):
    balls = []
    for p,f in zip(P, F):
        s = plt.Circle(p, f, **kwargs)
        balls.append(s)
        ax.add_patch(s)
    return balls

def plot_rips(ax, P, K, thresh, color, visible=True, dim=2, zorder=1):
    plot = {d : [] for d in range(2)}
    plot[2] = plot_poly(ax, P, K[2], visible, color=color, alpha=0.5, zorder=zorder+1)
    plot[1] = plot_edges(ax, P, K[1], visible, color=color, zorder=zorder+2, lw=1)
    # plot[0].set_visible(visible)
    return plot
