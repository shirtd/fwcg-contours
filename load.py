import numpy as np
import os, sys

from plot.surf import *
from rainier.data import *
from rainier.style import *
from rainier.util import *

SAVE = True

if __name__ == "__main__":
    fname = 'data/seattle16.dat' if len(sys.argv) < 2 else sys.argv[1]
    path, ext = os.path.splitext(fname)
    name = os.path.basename(path)

    key = "".join(filter(lambda c: not c.isdigit(), name))
    VIEW = VIEWS[key] if key in VIEWS else VIEWS['rainier']
    STRETCH = STRETCHES[key] if key in STRETCHES else STRETCH
    PAD = PADS[key] if key in PADS else PAD

    G = load_dat(fname, PAD if len(sys.argv) < 3 else int(sys.argv[2]))
    DIMS = (STRETCH, STRETCH * G.shape[1] / G.shape[0])
    X, Y = get_grid(G, DIMS)

    if SAVE:
        mlab.options.offscreen = True

    surf = SurfacePlot(X, Y, G, CUT_ARGS, SURF_ARGS, CONT_ARGS, VIEW)

    if SAVE:
        DIR = os.path.join('figures', '%s%s' % (name,(str(PAD) if PAD > 0 else '')))
        if not os.path.exists(DIR):
            os.makedirs(DIR)
        surf.render(DIR, LABELS)
