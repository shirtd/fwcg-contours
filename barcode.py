import dionysus as dio
import numpy as np
import os, sys

from rainier.util import *
from rainier.style import *
from rainier.data import *
from plot.persist import *

plt.ion()

THRESH = 5e-3

LINE_WIDTH = 1.5
FIGSIZE = (12,8)
DPI = 300

if __name__ == "__main__":
    fname = 'data/rainier-16.dat' if len(sys.argv) < 2 else sys.argv[1]
    path, ext = os.path.splitext(fname)
    name = os.path.basename(path)

    key = "".join(filter(lambda c: not c.isdigit(), name))
    PAD = PADS[key] if key in PADS else PAD

    G = load_dat(fname, PAD)
    X, Y = get_grid(G)

    mx, lim = G.max(), G.max()

    filt = dio.fill_freudenthal(G)
    hom = dio.homology_persistence(filt)
    dgms = dio.init_diagrams(hom, filt)

    np_dgms = [np.array([[p.birth, p.death if p.death < np.inf else -0.1] for p in d]) if len(d) else np.ndarray((0,2)) for d in dgms]
    ax = plot_barcode(np_dgms[1], CUT_ARGS, LINE_WIDTH, THRESH, figsize=FIGSIZE)
    plt.savefig('figures/%s_barcode.png' % name, dpi=DPI)
