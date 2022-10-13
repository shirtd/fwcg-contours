import numpy as np
import numpy.linalg as la
from itertools import combinations


def get_delta(n, w=1, h=1):
    return 2 / (n-1) * np.sqrt(w ** 2 + h ** 2)

def lipschitz(f, P):
    return max(abs(fp - fq) / la.norm(p - q) for (fp,p),(fq,q) in combinations(zip(f,P),2))
