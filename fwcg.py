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

import argparse

parser = argparse.ArgumentParser(prog='sample')

parser.add_argument('--dir', default=os.path.join('figures','lips_sub'), help='dir')
parser.add_argument('--file', default='data/surf_1194_1e-1.csv', help='file')
parser.add_argument('--sub', default='data/surf-sample_329_2e-1.csv', help='subsample file')
parser.add_argument('--dpi', type=int, default=300, help='dpi')
parser.add_argument('--save', action='store_true', help='save')
parser.add_argument('--mult', type=float, default=1., help='thresh mult')
parser.add_argument('--wait', type=float, default=0.5, help='wait')
parser.add_argument('--cmult', type=float, default=1., help='c mult')

plt.ion()

if __name__ == '__main__':
    args = parser.parse_args()

    G = mk_gauss(X, Y, GAUSS_ARGS)
    _c = args.cmult*3.1443048369350226 # lipschitz(G.flatten(), S)

    fig, ax = plt.subplots(figsize=(10,8))
    surf = ax.contourf(X, Y, G, levels=CUTS, colors=[COLOR[c] for c in COLORS], alpha=0, zorder=0)
    # contour = ax.contour(X, Y, G, levels=CUTS, colors=[COLOR[c] for c in COLORS], alpha=1, zorder=0)
    contour = ax.contour(X, Y, G, levels=CUTS, colors='black', alpha=1, zorder=0)
    ax.axis('off')
    ax.axis('scaled')
    ax.set_ylim(-2,2)
    ax.set_xlim(-3,3)
    plt.tight_layout()

    fname = args.file
    dir = os.path.dirname(fname)
    file = os.path.basename(fname)
    label, ext = os.path.splitext(file)
    lname = label.split('_')
    name, NPTS, THRESH = lname[0], lname[1], float(lname[2])

    sample = np.loadtxt(fname)
    subsample = np.loadtxt(args.sub)
    subname,_ = os.path.splitext(os.path.basename(args.sub))
    S, F = subsample[:,:2], subsample[:,2]
    P = np.vstack([sample[:,:2],S])
    points = ax.scatter(P[:,0], P[:,1], zorder=5, s=10, facecolors='none', edgecolors='black')
    points = ax.scatter(S[:,0], S[:,1], c='black', zorder=5, s=20)

    Fmin, Fmax = [], []
    for i,p in enumerate(P):
        Fmin.append(max(f - _c*la.norm(p - s) for s,f in zip(S,F)))
        Fmax.append(min(f + _c*la.norm(p - s) for s,f in zip(S,F)))

    K = rips(P, THRESH)
    max_plot = plot_rips(ax, P[:,:2], K, THRESH, COLOR['blue'], False, zorder=2)
    min_plot = plot_rips(ax, P[:,:2], K, THRESH, COLOR['red'], False, zorder=1)

    Emax = {e : max(Fmax[e[0]], Fmax[e[1]]) for e in K[1]}
    Emin = {e : max(Fmin[e[0]], Fmin[e[1]]) for e in K[1]}
    Tmax = {t : max(Emax[e] for e in combinations(t,2)) for t in K[2]}
    Tmin = {t : max(Emin[e] for e in combinations(t,2)) for t in K[2]}

    if args.save and not os.path.exists(args.dir):
        os.makedirs(args.dir)

    Fmin, Fmax = F.min(), F.max()
    levels = [Fmin-Fmax/2] + CUTS + [1.3*Fmax]
    for i, t in enumerate(levels):
        for s in K[2]:
            if Tmax[s] <= t:
                max_plot[2][s].set_visible(True)
            if Tmin[s] <= t:
                min_plot[2][s].set_visible(True)
        for s in K[1]:
            if Emax[s] <= t:
                max_plot[1][s].set_visible(True)
            if Emin[s] <= t:
                min_plot[1][s].set_visible(True)
        plt.pause(args.wait)
        if args.save:
            cmult_s = ('cx' + np.format_float_scientific(args.cmult, trim='-')) if int(args.cmult) != args.mult else ''
            plt.savefig(os.path.join(args.dir, '%s_subsample_lips%d%s.png' % (subname,i,cmult_s)), dpi=args.dpi)
