

def is_boundary(p, d, l):
    return not all(d < c < u - d for c,u in zip(p, l))

def to_path(vertices, nbrs):
    V = vertices.copy()
    cur = V.pop()
    path = [cur]
    while len(V):
        cur = nbrs[cur].intersection(V).pop()
        path.append(cur)
        V.remove(cur)
    return path
