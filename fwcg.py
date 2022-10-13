import numpy as np

from itertools import combinations
import numpy.linalg as la
from config.surf import *
from config.style import COLOR
from util.math import mk_gauss
from util.util import *
from util.geometry import rips
from plot.mpl import *
import os, sys
from scipy.spatial import KDTree

import dionysus as dio

plt.ion()

DPI = 300
DIR = os.path.join('figures','lips')
SAVE = True
WAIT = 0.5
DO_MIN = True
DO_MAX = True

if __name__ == '__main__':
    G = mk_gauss(X, Y, GAUSS_ARGS)
    _c = 3.1443048369350226 # lipschitz(G.flatten(), S)

    fig, ax = plt.subplots(figsize=(10,8))
    surf = ax.contourf(X, Y, G, levels=CUTS, colors=[COLOR[c] for c in COLORS], alpha=0.5, zorder=0)
    contour = ax.contour(X, Y, G, levels=CUTS, colors=[COLOR[c] for c in COLORS], alpha=1, zorder=0)
    ax.axis('off')
    ax.axis('scaled')
    ax.set_ylim(-2,2)
    ax.set_xlim(-3,3)
    plt.tight_layout()

    fname = 'data/surf-sample_329_1e-01.csv' if len(sys.argv) < 2 else sys.argv[1]
    dir = os.path.dirname(fname)
    file = os.path.basename(fname)
    label, ext = os.path.splitext(file)
    lname = label.split('_')
    name, NPTS, THRESH = lname[0], lname[1], 2*float(lname[2])

    sample = np.loadtxt(fname)
    subsample = np.loadtxt('data/surf-sample_43_1e-01.csv')
    P = np.vstack([sample[:,:2], subsample[:,:2]])
    S, F = subsample[:,:2], subsample[:,2]
    points = ax.scatter(P[:,0], P[:,1], zorder=5, s=10, facecolors='none', edgecolors='black')
    points = ax.scatter(S[:,0], S[:,1], c='black', zorder=5, s=20)

    Fmin, Fmax = [], []
    for i,p in enumerate(P):
        Fmin.append(max(f - _c*la.norm(p - s) for s,f in zip(S,F)))
        Fmax.append(min(f + _c*la.norm(p - s) for s,f in zip(S,F)))

    plt.savefig(os.path.join('figures', 'lips', 'surf-sample_43_1e-01_both.png'), dpi=DPI)


    # K = rips(P, THRESH)
    # max_plot = plot_rips(ax, P[:,:2], K, THRESH, COLOR['blue'], False, zorder=2)
    # min_plot = plot_rips(ax, P[:,:2], K, THRESH, COLOR['red'], False, zorder=1)
    #
    # Emax = {e : max(Fmax[e[0]], Fmax[e[1]]) for e in K[1]}
    # Emin = {e : max(Fmin[e[0]], Fmin[e[1]]) for e in K[1]}
    # Tmax = {t : max(Emax[e] for e in combinations(t,2)) for t in K[2]}
    # Tmin = {t : max(Emin[e] for e in combinations(t,2)) for t in K[2]}
    #
    # Fmin, Fmax = F.min(), F.max()
    # levels = [Fmin-Fmax/2] + CUTS + [1.3*Fmax]
    # # for i, t in enumerate(np.linspace(Fmin-Fmx/2, 1.3*Fmx, N)):
    # for i, t in enumerate(levels):
    #     for s in K[2]:
    #         if DO_MAX and Tmax[s] <= t:
    #             max_plot[2][s].set_visible(True)
    #         if DO_MIN and Tmin[s] <= t:
    #             min_plot[2][s].set_visible(True)
    #     for s in K[1]:
    #         if DO_MAX and Emax[s] <= t:
    #             max_plot[1][s].set_visible(True)
    #         if DO_MIN and Emin[s] <= t:
    #             min_plot[1][s].set_visible(True)
    #     plt.pause(WAIT)
    #     if SAVE:
    #         plt.savefig(os.path.join(DIR, '%s_lips_sub_tri%d.png' % (label, i)), dpi=DPI)
