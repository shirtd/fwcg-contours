import matplotlib.pyplot as plt
import numpy as np
import util.util
# plt.ion()

def plot_poly(ax, T, *args, **kwargs):
    tp = [plt.Polygon(t, *args, **kwargs) for t in T]
    return util.util.lmap(lambda t: ax.add_patch(t), tp)

def plot_edges(ax, E, *args, **kwargs):
    return [e for e in [ax.plot(e[:,0], e[:,1], *args, **kwargs) for e in E]]
