from functools import reduce

def lmap(f, l):
    return list(map(f,l))

def format_float(f):
    if f.is_integer():
        return int(f)
    e = 0
    while not f.is_integer():
        f *= 10
        e -= 1
    return '%de%d' % (int(f), e)

def insert(L, i, x):
    L[i] += [x]
    return L

def partition(f, X, n):
    return reduce(f, X, [[] for _ in range(n)])

def diff(p):
    return p[1] - p[0]

def identity(x):
    return x

def stuple(s, *args, **kw):
    return tuple(sorted(s, *args, **kw))

def grid_coord(v, N):
    return [v//N, v%N]
