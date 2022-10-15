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
import os, sys
from scipy.spatial import KDTree

import dionysus as dio

SEED = np.random.randint(10000000) # 1584326
print('seed: %d' % SEED)
np.random.seed(SEED)

plt.ion()

DPI = 300
DIR = os.path.join('figures','lips')
SAVE = False
WAIT = 0.5
DO_MIN = True
DO_MAX = True

def plot_barcode(ax, dgm, cuts, lw=5, thresh=0, *args, **kwargs):
    dgm = np.array([p for p in dgm if p[1]-p[0] > thresh and p[1] != np.inf])
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
            # if death == np.inf:
            #       ax.plot([lim, lim+0.1], [i, i], c='black', linestyle='dotted')
    ax.get_yaxis().set_visible(False)
    plt.tight_layout()
    return ax

if __name__ == '__main__':

    G = mk_gauss(X, Y, GAUSS_ARGS)
    _c = 3.1443048369350226 # lipschitz(G.flatten(), S)

    # fname = 'data/surf-sample_329_1e-01.csv' if len(sys.argv) < 2 else sys.argv[1]
    fname = 'data/surf-sample_396_1e-01.csv' if len(sys.argv) < 2 else sys.argv[1]
    dir = os.path.dirname(fname)
    file = os.path.basename(fname)
    label, ext = os.path.splitext(file)
    lname = label.split('_')
    name, NPTS, THRESH = lname[0], lname[1], 0.045 # 2*float(lname[2])

    sample = G # np.loadtxt(fname)
    # subsample = np.loadtxt('data/surf-sample_107_2e-01.csv')
    subsample = np.loadtxt(fname)
    P = np.vstack([sample[:,:2], subsample[:,:2]])
    # idx = np.random.randint(0,len(sample),300)
    # subsample = sample[idx]
    P, _F = sample[:,:2], sample[:,2]
    S, F = subsample[:,:2], subsample[:,2]

    Fmin, Fmax = [], []
    for i,p in enumerate(P):
        Fmin.append(max(f - _c*la.norm(p - s) for s,f in zip(S,F)))
        Fmax.append(min(f + _c*la.norm(p - s) for s,f in zip(S,F)))

    Fmin = np.array(Fmin)
    Fmax = np.array(Fmax)

    K = RipsComplex(P, 2*THRESH)
    # induce(K, Q_FUN, 'fun', 'max')
    for s in K:
        s.data['min'] = max(Fmin[s])
        s.data['max'] = max(Fmax[s]) if s.data['dist'] <= THRESH else np.inf


    min_filt = Filtration(K, 'min')
    max_filt = Filtration(K, 'max')
    hom =  Diagram(K, min_filt, pivot=max_filt)
    dgm,_ = hom.get_diagram(K, min_filt, max_filt)

    filt = dio.fill_freudenthal(G)
    hom = dio.homology_persistence(filt)
    dgms = dio.init_diagrams(hom, filt)
    np_dgms = [np.array([[p.birth, p.death if p.death < np.inf else -0.1] for p in d]) if len(d) else np.ndarray((0,2)) for d in dgms]

    fig, ax = plt.subplots(2,1,sharex=True, sharey=True,figsize=(6,4))
    ax[0].invert_yaxis()
    plt.tight_layout()

    plot_barcode(ax[0], dgm[1], CUT_ARGS)
    plot_barcode(ax[1], np_dgms[1], CUT_ARGS)

    ax[1].set_xticks([], [])
    ax[1].set_ylim(5,-1)

    # plt.savefig('figures/surf-sample_329_1e-01_ripslips.png', dpi=300)

    if input("save?"):
        fname = 'figures/surf_subsample_107_2e-01_lips.png'
        print('saving %s' % fname)
        plt.savefig(fname, dpi=300)
