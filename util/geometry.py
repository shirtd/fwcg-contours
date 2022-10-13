import numpy as np
import numpy.linalg as la
from itertools import combinations
from tqdm import tqdm
import dionysus as dio
from util.util import stuple


def get_delta(n, w=1, h=1):
    return 2 / (n-1) * np.sqrt(w ** 2 + h ** 2)

def lipschitz(F, P):
    return max(abs(fp - fq) / la.norm(p - q) for (fp,p), (fq,q) in tqdm(list(combinations(zip(F,P), 2))))

def rips(P, thresh, dim=2):
    K = {d : [] for d in range(dim+1)}
    S = dio.fill_rips(P, dim, thresh)
    for s in S:
        K[s.dimension()].append(stuple(s))
    return K
