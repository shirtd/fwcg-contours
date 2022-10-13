import numpy as np
import tqdm


def get_grid(G, dims=None):
    dims = G.shape if dims is None else dims
    X_RNG = np.linspace(-dims[0]/2, dims[0]/2, G.shape[0])
    Y_RNG = np.linspace(-dims[1]/2, dims[1]/2, G.shape[1])
    return np.meshgrid(X_RNG, Y_RNG)

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
    for j in tqdm.tqdm(range(_M)):
        for i in range(_N):
            x = G[i*l:(i+1)*l, j*l:(j+1)*l].sum() / (l ** 2)
            D[i, j] = x if x > 0 else 0
    return D
