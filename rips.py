import numpy as np

from config.surf import *
from config.style import COLOR
from util.math import mk_gauss
from util.util import *
from plot.mpl import *
import os, sys

import dionysus as dio


plt.ion()

def rips(P, thresh, dim=2):
    K = {d : [] for d in range(dim+1)}
    S = dio.fill_rips(P, dim, thresh)
    for s in S:
        K[s.dimension()].append(stuple(s))
    return K

def plot_rips(ax, P, K, thresh, dim=2):
    plot = {d : [] for d in range(2)}
    plot[2] = plot_poly(ax, P[K[2]], color=COLOR['red'], alpha=0.5, zorder=1)
    plot[1] = plot_edges(ax, P[K[1]], color='black', zorder=3, lw=1)
    plot[0] = ax.scatter(P[:,0], P[:,1], c='black', zorder=4, s=10)
    return plot

if __name__ == '__main__':
    G = mk_gauss(X, Y, GAUSS_ARGS)

    fig, ax = plt.subplots(figsize=(10,8))
    surf = ax.contourf(X, Y, G, levels=CUTS, colors=[COLOR[c] for c in COLORS], alpha=0.5, zorder=0)
    contour = ax.contour(X, Y, G, levels=CUTS, colors=[COLOR[c] for c in COLORS], zorder=0)
    ax.axis('off')
    ax.axis('scaled')

    fname = 'data/surf-sample_329_1e-01.csv' if len(sys.argv) < 2 else sys.argv[1]
    dir = os.path.dirname(fname)
    file = os.path.basename(fname)
    label, ext = os.path.splitext(file)
    lname = label.split('_')
    name, NPTS, THRESH = lname[0], lname[1], 2*float(lname[2])

    sample = np.loadtxt(fname)
    P, F = sample[:,:2], sample[:,2]
    K = rips(P, THRESH)
    plot = plot_rips(ax, P, K, THRESH)
