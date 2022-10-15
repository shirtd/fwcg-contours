from topology.complex.simplicial import *
from topology.complex.cellular import *
from topology.complex.geometric import *

from topology.persistence.filtration import *
from topology.persistence.diagram import *
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

import ripser

import dionysus as dio

import argparse

parser = argparse.ArgumentParser(prog='sample')

parser.add_argument('--dir', default='figures', help='dir')
parser.add_argument('--file', default='data/surf_1194_1e-1.csv', help='file')
parser.add_argument('--sub', default='data/surf-sample_329_2e-1.csv', help='subsample file')
parser.add_argument('--dpi', type=int, default=300, help='dpi')
parser.add_argument('--save', action='store_true', help='save')
parser.add_argument('--mult', type=float, default=1., help='thresh mult')
parser.add_argument('--cmult', type=float, default=1., help='lipschitz constant mult')

# plt.ion()

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

    args = parser.parse_args()

    G = mk_gauss(X, Y, GAUSS_ARGS)
    _c = args.cmult*3.1443048369350226 # lipschitz(G.flatten(), S)

    fname = args.file
    dir = os.path.dirname(fname)
    file = os.path.basename(fname)
    label, ext = os.path.splitext(file)
    lname = label.split('_')
    name, NPTS, THRESH = lname[0], lname[1], float(lname[2])

    sample = np.loadtxt(fname)
    subsample = np.loadtxt(args.sub)
    subname,_ = os.path.splitext(os.path.basename(args.sub))
    P, _F = sample[:,:2], sample[:,2]
    S, F = subsample[:,:2], subsample[:,2]

    Fmin, Fmax = [], []
    for i,p in enumerate(P):
        Fmin.append(max(f - _c*la.norm(p - s) for s,f in zip(S,F)))
        Fmax.append(min(f + _c*la.norm(p - s) for s,f in zip(S,F)))

    Fmin = np.array(Fmin)
    Fmax = np.array(Fmax)

    K = RipsComplex(P, args.mult*THRESH)
    for s in K:
        s.data['min'] = max(Fmin[s])
        s.data['max'] = max(Fmax[s]) if s.data['dist'] <= THRESH else np.inf

    min_filt = Filtration(K, 'min')
    max_filt = Filtration(K, 'max')
    hom =  Diagram(K, min_filt, pivot=max_filt, verbose=True)
    dgm,_ = hom.get_diagram(K, min_filt, max_filt)

    fig, ax = plt.subplots(2,1,sharex=True, sharey=True,figsize=(6,4))
    ax[0].invert_yaxis()
    plt.tight_layout()

    plot_barcode(ax[0], dgm[1], CUT_ARGS)

    filt = dio.fill_freudenthal(G)
    hom = dio.homology_persistence(filt)
    dgms = dio.init_diagrams(hom, filt)
    np_dgms = [np.array([[p.birth, p.death if p.death < np.inf else -0.1] for p in d]) if len(d) else np.ndarray((0,2)) for d in dgms]

    plot_barcode(ax[1], np_dgms[1], CUT_ARGS)

    ax[1].set_xticks([], [])
    ax[1].set_ylim(5,-1)

    if args.save and not os.path.exists(args.dir):
        os.makedirs(args.dir)

    if args.save:
        mult_s = np.format_float_scientific(args.mult, trim='-') if int(args.mult) != args.mult else str(int(args.mult))
        cmult_s = ('cx' + np.format_float_scientific(args.cmult, trim='-')) if int(args.cmult) != args.mult else ''
        fname = os.path.join(args.dir,'%s_subsample_lips%s%s.png' % (subname,mult_s,cmult_s))
        print('saving %s' % fname)
        plt.savefig(fname, dpi=args.dpi)
        plt.show()
