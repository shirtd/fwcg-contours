from topology.complex.simplicial import *
from topology.complex.cellular import *
from topology.complex.geometric import *

from topology.persistence.filtration import *
from topology.persistence.diagram import *

from topology.util import lipschitz, dio_diagram, pmap
from topology.plot.util import plot_diagrams
from topology.plot.lipschitz import *
from topology.data import *

import numpy as np

from itertools import combinations
import numpy.linalg as la
from config.surf import *
from config.style import COLOR
from util.math import mk_gauss
from util.util import *
from util.geometry import rips
from plot.mpl import *
from plot.persist import *
import os, sys

import dionysus as dio

plt.ion()

DPI = 300
DIR = os.path.join('figures','lips')
SAVE = True
WAIT = 0.5
DO_MIN = True
DO_MAX = True

def induce(K, f, key, finduce=max):
    for s in K:
        s.data[key] = finduce(f[i] for i in s)

def max_ext(Q, fQ, l, d, p):
    # return min(f + l*(la.norm(p - q) - d) for q,f in zip(Q, fQ))
    return min(f + l*(la.norm(p - q)) for q,f in zip(Q, fQ))

def min_ext(Q, fQ, l, d, p):
    # return max(f - l*(la.norm(p - q) + d) for q,f in zip(Q, fQ))
    # return max(f - l*(la.norm(p - q) + d) for q,f in zip(Q, fQ))
    return max(f - l*(la.norm(p - q)) for q,f in zip(Q, fQ))

def minmax_ext(Q, fQ, l, d, p):
    return [e(Q, fQ, l, d, p) for e in (max_ext, min_ext)]

def lipschitz_extend(K, Q, fQ, l, d=0, finduce=max, verbose=True):
    it = tqdm(F.P, desc='[ min/max ext') if verbose else F.P
    # es = list(zip(*[[e(Q, fQ, l, d, p) for e in (max_ext, min_ext)] for p in it]))
    es = list(zip(*pmap(minmax_ext, it, Q, fQ, l, d)))
    for e,k in zip(es, ('maxext', 'minext')):
        induce(F, e, k, finduce)
    return es

def plot_rips(ax, P, K, thresh, color, visible=True, dim=2, zorder=1):
    plot = {d : [] for d in range(2)}
    plot[2] = plot_poly(ax, P, K(2), visible, color=color, alpha=0.5, zorder=zorder+1)
    plot[1] = plot_edges(ax, P, K(1), visible, color=color, zorder=zorder+2, lw=1)
    # plot[0].set_visible(visible)
    return plot

def plot_barcode(ax, dgm, cuts, lw=5, thresh=0, *args, **kwargs):
    dgm = np.array([p for p in dgm if p[1]-p[0] > thresh])
    if not len(dgm):
        return None
    for i, (birth, death) in enumerate(dgm):
        for name, v in cuts.items():
            a, b, c = v['min'], v['max'], v['color']
            if a < birth and death <= b:
                ax.plot([birth, death], [i, i], c=c, lw=lw)
            elif birth < a and death > a and death <= b:
                ax.plot([a, death], [i, i], c=c, lw=lw)
            elif birth > a and birth < b and death > b:
                ax.plot([birth, b], [i, i], c=c, lw=lw)
            elif birth <= a and b < death:
                ax.plot([b, a], [i, i], c=c, lw=lw)
        if death == np.inf:
            ax.plot([lim, lim+0.1], [i, i], c='black', linestyle='dotted')
    ax.get_yaxis().set_visible(False)
    plt.tight_layout()
    return ax

if __name__ == '__main__':
    G = mk_gauss(X, Y, GAUSS_ARGS)
    _c = 3.1443048369350226 # lipschitz(G.flatten(), S)
    #
    # fig, ax = plt.subplots(figsize=(10,8))
    # surf = ax.contourf(X, Y, G, levels=CUTS, colors=[COLOR[c] for c in COLORS], alpha=0, zorder=0)
    # contour = ax.contour(X, Y, G, levels=CUTS, colors=[COLOR[c] for c in COLORS], alpha=0, zorder=0)
    # ax.axis('off')
    # ax.axis('scaled')
    # ax.set_ylim(-2,2)
    # ax.set_xlim(-3,3)
    # plt.tight_layout()
    #
    # fname = 'data/surf-sample_329_1e-01.csv' if len(sys.argv) < 2 else sys.argv[1]
    # dir = os.path.dirname(fname)
    # file = os.path.basename(fname)
    # label, ext = os.path.splitext(file)
    # lname = label.split('_')
    # name, NPTS, THRESH = lname[0], lname[1], 2*float(lname[2])
    #
    # sample = np.loadtxt(fname)
    # P, F = sample[:,:2], sample[:,2]
    # points = ax.scatter(P[:,0], P[:,1], c='black', zorder=5, s=10)
    #
    # K = RipsComplex(P, 2*THRESH)
    # # induce(K, Q_FUN, 'fun', 'max')
    # for s in K:
    #     s.data['fun'] = max(F[s])
    #     s.data['fun0'] = s.data['fun'] if s.data['dist'] <= THRESH else np.inf
    #
    # plot = plot_rips(ax, P, K, THRESH, COLOR['red'], False)
    # filt = Filtration(K, 'fun')
    # filt0 = Filtration(K, 'fun0')
    # hom =  Diagram(K, filt, pivot=filt0)
    # dgm = hom.get_diagram(K, filt, filt0)

    filt = dio.fill_freudenthal(G)
    hom = dio.homology_persistence(filt)
    dgms = dio.init_diagrams(hom, filt)
    np_dgms = [np.array([[p.birth, p.death if p.death < np.inf else -0.1] for p in d]) if len(d) else np.ndarray((0,2)) for d in dgms]
    sub = np.array([[0.1054815 , 1.13615359],[0.41517061, 0.62644626]])
    # ax = plot_barcode(dggm, CUT_ARGS, 5, 0, figsize=(6,4))

    fig, ax = plt.subplots(2,1,sharex=True, sharey=True,figsize=(6,4))
    ax[0].invert_yaxis()
    plt.tight_layout()

    plot_barcode(ax[0], sub, CUT_ARGS)
    plot_barcode(ax[1], np_dgms[1], CUT_ARGS)

    ax[1].set_xticks([], [])
    ax[1].set_ylim(5,-1)

    plt.savefig('figures/surf-sample_329_1e-01_sfa.png', dpi=300)




    # CACHE_PATH = '%s_%s.pkl' % (LABEL, FCACHE)
    # if not FORCE and LOAD and os.path.exists(CACHE_PATH):
    #     print('loading %s' % CACHE_PATH)
    #     DQ, D, VQ, V, B, delta, filt, hom, Ddio = pkl.load(open(CACHE_PATH, 'rb'))
    # else:
    #     DQ, D = DelaunayComplex(Q), DelaunayComplex(P)
    #     VQ, V = VoronoiComplex(DQ), VoronoiComplex(D)
    #     B = D.get_boundary(BOUNDS)
    #     delta = max(max(la.norm(D.P[s[0]] - V.P[j]) for j in V.dual(s)) for s in D(0) if not s in B)
    #     induce(DQ, VQ, Q_FUN, 'fun', INDUCE)  # max) #
    #     induce(D, V, P_FUN, 'fun', INDUCE) # max) #
    #     lipschitz_extend(D, V, Q, Q_FUN, lips, delta, INDUCE) # max) #
    #
    #
    #
    #     if LABEL == 'voronoi':
    #         filt = {'Qfun' : Filtration(VQ, 'fun'),
    #                 'Pfun' : Filtration(V, 'fun'),
    #                 'maxext' : Filtration(V, 'maxext'),
    #                 'minext' : Filtration(V, 'minext', False)}
    #
    #         hom = {'Qfun' : Diagram(VQ, filt['Qfun']),
    #                 'Pfun' : Diagram(V, filt['Pfun']),
    #                 'maxext' : Diagram(V, filt['maxext']),
    #                 'minext' : Diagram(V, filt['minext']),
    #                 'image' : Diagram(V, filt['minext'], pivot=filt['maxext'])}
    #
    #     elif LABEL == 'delaunay':
    #         filt = {'Qfun' : Filtration(DQ, 'fun'),
    #                 'Pfun' : Filtration(D, 'fun'),
    #                 'maxext' : Filtration(D, 'maxext'),
    #                 'minext' : Filtration(D, 'minext', False)}
    #
    #         hom = {'Qfun' : Diagram(DQ, filt['Qfun']),
    #                 'Pfun' : Diagram(D, filt['Pfun']),
    #                 'maxext' : Diagram(D, filt['maxext']),
    #                 'minext' : Diagram(D, filt['minext']),
    #                 'image' : Diagram(D, filt['minext'], pivot=filt['maxext'])}
