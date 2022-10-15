import numpy as np
import numpy.linalg as la
from util.math import mk_gauss
from config.surf import *


if __name__ == '__main__':
    fun = mk_gauss(X, Y, GAUSS_ARGS).flatten()
    idx = [i for i,f in enumerate(fun) if f > CUTS[0]]
    points = np.vstack([X.flatten()[idx], Y.flatten()[idx], fun[idx]]).T
    delta = la.norm(np.array([X[0,0],Y[0,0]]) - np.array([X[1,1],Y[1,1]]))
    np.savetxt('data/surf_%d_%s' % (len(points), np.format_float_scientific(delta, trim='-')), points)
