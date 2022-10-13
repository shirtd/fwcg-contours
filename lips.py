import numpy as np

from config.surf import *
from config.style import COLOR
from util.math import mk_gauss
from util.geometry import lipschitz, rips
from itertools import combinations
from util.util import *
from plot.mpl import *
import os, sys

import dionysus as dio

plt.ion()

DPI = 300
DIR = os.path.join('figures','lips')
SAVE = True
WAIT = 0.5
DO_MIN = True
DO_MAX = False

if __name__ == '__main__':
    G = mk_gauss(X, Y, GAUSS_ARGS)
    S = np.vstack([X.flatten(),Y.flatten()]).T
    _c = 3.1443048369350226 # lipschitz(G.flatten(), S)
    # print(_c)

    fig, ax = plt.subplots(figsize=(10,8))
    surf = ax.contourf(X, Y, G, levels=CUTS, colors=[COLOR[c] for c in COLORS], alpha=0, zorder=0)
    contour = ax.contour(X, Y, G, levels=CUTS, colors=[COLOR[c] for c in COLORS], alpha=0, zorder=0)
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
    P, F = sample[:,:2], sample[:,2]
    points = ax.scatter(P[:,0], P[:,1], c='black', zorder=5, s=10)
    balls_max = plot_balls(ax, P, 2 * F/_c, color=COLOR['blue'], zorder=4, alpha=0.3 if DO_MAX else 0)
    balls_min = plot_balls(ax, P, 2 * F/_c, color=COLOR['red'], zorder=4, alpha=0.3 if DO_MIN else 0)

    if SAVE and not os.path.exists(DIR):
        os.makedirs(DIR)

    Fmin, Fmax = F.min(), F.max()
    levels = [Fmin-Fmax/2] + CUTS + [1.3*Fmax]
    # for i, t in enumerate(np.linspace(Fmin-Fmax/2, 1.3*Fmax, N)):
    for i, t in enumerate(levels):
        for f, mn, mx in zip(F, balls_min, balls_max):
            fmax = (t - f) / _c
            fmin = (f - t) / _c
            mn.set_radius(fmin if fmin > 0 else 0)
            mx.set_radius(fmax if fmax > 0 else 0)
        plt.pause(WAIT)
        if SAVE:
            tag = 'min' if not DO_MAX else 'max' if not DO_MIN else ''
            plt.savefig(os.path.join(DIR, '%s_lips%s%d.png' % (label, tag, i)), dpi=DPI)
