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

import dionysus as dio

parser = argparse.ArgumentParser(prog='sample')

parser.add_argument('--dir', default=os.path.join('figures','lips'), help='dir')
parser.add_argument('--file', default='data/surf-sample_329_1e-01.csv', help='file')
parser.add_argument('--dpi', type=int, default=300, help='dpi')
parser.add_argument('--wait', type=float, default=0.5, help='wait')
parser.add_argument('--save', action='store_true', help='save')
parser.add_argument('--no-min', action='store_false', help='max only')
parser.add_argument('--no-max', action='store_false', help='min only')

plt.ion()

if __name__ == '__main__':
    DPI = args.dpi
    DIR = args.dir
    SAVE = args.save
    WAIT = args.wait
    DO_MIN = args.no_min
    DO_MAX = args.no_max

    G = mk_gauss(X, Y, GAUSS_ARGS)
    _c = 3.1443048369350226 # lipschitz(G.flatten(), S)

    fig, ax = plt.subplots(figsize=(10,8))
    surf = ax.contourf(X, Y, G, levels=CUTS, colors=[COLOR[c] for c in COLORS], alpha=0, zorder=0)
    # contour = ax.contour(X, Y, G, levels=CUTS, colors=[COLOR[c] for c in COLORS], alpha=1, zorder=0)
    contour = ax.contour(X, Y, G, levels=CUTS, colors='black', alpha=1, zorder=0)
    ax.axis('off')
    ax.axis('scaled')
    ax.set_ylim(-2,2)
    ax.set_xlim(-3,3)
    plt.tight_layout()

    # fname = 'data/surf-sample_329_1e-01.csv' if len(sys.argv) < 2 else sys.argv[1]
    fname = 'data/surf_2048_1e-1.csv' if len(sys.argv) < 2 else sys.argv[1]
    fname = 'data/surf_1194_1e-1.csv' if len(sys.argv) < 2 else sys.argv[1]
    # fname = 'data/surf_279_2e-1.csv' if len(sys.argv) < 2 else sys.argv[1]
    dir = os.path.dirname(fname)
    file = os.path.basename(fname)
    label, ext = os.path.splitext(file)
    lname = label.split('_')
    name, NPTS, THRESH = lname[0], lname[1], float(lname[2])

    sample = np.loadtxt(fname)
    P, F = sample[:,:2], sample[:,2]
    points = ax.scatter(P[:,0], P[:,1], c='black', zorder=5, s=10)

    K = rips(P, THRESH)
    max_plot = plot_rips(ax, P[:,:2], K, THRESH, COLOR['blue'], False)
    min_plot = plot_rips(ax, P[:,:2], K, THRESH, COLOR['red'], True)

    Edist = {e : la.norm(P[e[0]] - P[e[1]]) for e in K[1]}
    Emax = {e : (F[e[0]]+F[e[1]] + _c * Edist[e]) / 2 for e in K[1]}
    Emin = {e : (F[e[0]]+F[e[1]] - _c * Edist[e]) / 2 for e in K[1]}
    Tmax = {t : max(Emax[e] for e in combinations(t,2)) for t in K[2]}
    Tmin = {t : min(Emin[e] for e in combinations(t,2)) for t in K[2]}

    Fmin, Fmax = F.min(), F.max()
    levels = [Fmin-Fmax/2] + CUTS + [1.3*Fmax]
    # for i, t in enumerate(np.linspace(Fmin-Fmx/2, 1.3*Fmx, N)):
    for i, t in enumerate(levels):
        for s in K[2]:
            if DO_MAX and Tmax[s] <= t:
                max_plot[2][s].set_visible(True)
            if DO_MIN and Tmin[s] <= t:
                min_plot[2][s].set_visible(False)
        for s in K[1]:
            if DO_MAX and Emax[s] <= t:
                max_plot[1][s].set_visible(True)
            if DO_MIN and Emin[s] <= t:
                min_plot[1][s].set_visible(False)
        plt.pause(WAIT)
        if SAVE:
            plt.savefig(os.path.join(DIR, '%s_lips_tri%d.png' % (label, i)), dpi=DPI)
