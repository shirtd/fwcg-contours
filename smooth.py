from tqdm import tqdm
import numpy as np
import sys, os

def down_sample(G, l):
    N, M = G.shape
    _N, nrem = divmod(N, l)
    _M, mrem = divmod(M, l)
    if nrem > 0 and mrem > 0:
        G = G[nrem//2:-nrem//2, mrem//2:-mrem//2]
    elif nrem > 0:
        G = G[nrem//2:-nrem//2, :]
    elif mrem > 0:
        G = G[:, mrem//2:-mrem//2]
    D = np.zeros((_N, _M), dtype=float)
    for j in tqdm(range(_M)):
        for i in range(_N):
            x = G[i*l:(i+1)*l, j*l:(j+1)*l].sum() / (l ** 2)
            D[i, j] = x if x > 0 else 0
    return D

if __name__ == '__main__':
    fname = 'data/output_USGS10m_rainier_small.asc' if len(sys.argv) < 2 else sys.argv[1]
    K = 4 if len(sys.argv) < 3 else int(sys.argv[2])
    
    name, ext = os.path.splitext(fname)
    dat = np.loadtxt(fname, skiprows=6 if ext == '.asc' else 0)
    smoothed = down_sample(dat, K)
    np.savetxt("%s%d.dat" % (name, K), smoothed, fmt="%f")
