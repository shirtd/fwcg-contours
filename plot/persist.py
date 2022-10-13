import matplotlib.pyplot as plt
import numpy.linalg as la
import numpy as np

def plot_barcode(dgm, cuts, lw=1, thresh=1e-2, *args, **kwargs):
    dgm = np.array([p for p in dgm if p[1]-p[0] > thresh])
    if not len(dgm):
        return None
    fig, ax = plt.subplots(1, 1, *args, **kwargs)
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
        if death == np.inf:
            ax.plot([lim, lim+0.1], [i, i], c='black', linestyle='dotted')
    ax.get_yaxis().set_visible(False)
    plt.tight_layout()
    return ax
