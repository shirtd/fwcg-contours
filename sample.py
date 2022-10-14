import argparse
import numpy as np

from config.surf import *
from config.style import COLOR
from util.math import mk_gauss
from plot.mpl import *
import os, sys

from scipy.spatial import KDTree

import dionysus as dio

parser = argparse.ArgumentParser(prog='sample')

parser.add_argument('--dataset', default='lennard-jones', help='data set')
parser.add_argument('--dir', default='data', help='dir')
parser.add_argument('--file', default=None, help='file')
parser.add_argument('--name', default='surf', help='name')
parser.add_argument('--load', default='data/surf-sample_329_1e-01.csv', help='view other')

parser.add_argument('--seed', type=int, default=None, help='seed')
parser.add_argument('--thresh', type=float, default=1e-1, help='radius')

MARGIN = 0 # THRESH/2
LEVELS = 20 # CUTS
COLORS = None # [COLOR[c] for c in COLORS]

def sample(fig, ax, S, thresh, color=COLOR['red'], name='surf-sample', dir='data'):
    P, T = [], KDTree(S[:,:2])
    def onclick(event):
        p = S[T.query(np.array([event.xdata,event.ydata]))[1]]
        ax.add_patch(plt.Circle(p, thresh, color=color, alpha=0.5, zorder=2))
        ax.scatter(p[0], p[1], c='black', zorder=3, s=10)
        plt.pause(0.1)
        P.append(p)
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    fig.canvas.mpl_disconnect(cid)
    return np.vstack(P)


if __name__ == '__main__':
    args = parser.parse_args()
    SEED = np.random.randint(10000000) if args.seed is None else args.seed
    print('seed: %d' % SEED)
    np.random.seed(SEED)

    G = mk_gauss(X, Y, GAUSS_ARGS)
    S = np.vstack([X.flatten(),Y.flatten(),G.flatten()]).T

    fig, ax = plt.subplots(figsize=(10,8))
    surf = ax.contourf(X, Y, G, levels=LEVELS, colors=COLORS, alpha=0.0, zorder=0)
    contour = ax.contour(X, Y, G, levels=LEVELS, colors=COLORS, zorder=0)
    ax.axis('off')
    ax.axis('scaled')
    ax.set_ylim(-2,2)
    ax.set_xlim(-3,3)
    plt.tight_layout()

    # if args.file is not None:
    #     P = np.loadtxt(args.file)

    if args.load is not None:
        ss = np.loadtxt(args.load)
        PP, F = ss[:,:2], ss[:,2]
        ppoints = ax.scatter(PP[:,0], PP[:,1], c='black', zorder=5, s=10, facecolors='none')
        bballs = plot_balls(ax, PP, np.ones(len(PP))*args.thresh, color=COLOR['blue'], zorder=4, alpha=0.3)

    P = sample(fig, ax, S, args.thresh)
    thresh_s = np.format_float_scientific(args.thresh, trim='-')
    name = '%s-sample' % args.name
    fname = os.path.join(args.dir, '%s_%d_%s.csv' % (name, len(P), thresh_s))
    if input('save %s (y/*)? ' % fname) in {'y','Y','yes'}:
        print('saving %s' % fname)
        np.savetxt(fname, P)
