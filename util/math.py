from multiprocessing import Pool
from tqdm import tqdm
import sys, time, gc
import pickle as pkl

import numpy as np
import scipy as sp


def gaussian(X, Y, c=[0., 0.], s=[0.5, 0.5]):
    return np.exp(-((X-c[0])**2 / (2*s[0]**2) + (Y-c[1])**2 / (2*s[1]**2)))

def make_gaussian(X, Y, args):
    return sum(w*gaussian(X, Y, c, r) for w, c, r in args)

def gaussian_random_field(alpha=-3.0, m=128, normalize=True):
    size = int(np.sqrt(m))
    k_ind = np.mgrid[:size, :size] - int((size + 1) / 2)
    k_idx = sp.fftpack.fftshift(k_ind)
    # Defines the amplitude as a power law 1/|k|^(alpha/2)
    amplitude = np.power(k_idx[0] ** 2 + k_idx[1] ** 2 + 1e-10, alpha / 4.0)
    amplitude[0,0] = 0
    # Draws a complex gaussian random noise with normal (circular) distribution
    noise = np.random.normal(size = (size, size)) + 1j * np.random.normal(size = (size, size))
    G = np.fft.ifft2(noise * amplitude).real # To real space
    return util.stats.scale(G) if normalize else G


mk_gauss = make_gaussian
grf = gaussian_random_field
