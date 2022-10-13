import numpy as np

CUTS = [0.05, 0.3, 0.55, 0.8, 1.3]
COLORS = ['green', 'blue', 'purple', 'yellow']

N, WIDTH, HEIGHT = 64, 2, 1
X_RNG = np.linspace(-WIDTH,WIDTH,WIDTH*N)
Y_RNG = np.linspace(-HEIGHT,HEIGHT,HEIGHT*N)
X, Y = np.meshgrid(X_RNG, Y_RNG)
GAUSS_ARGS = [  (1, [-0.2, 0.2], [0.3, 0.3]),
                (0.5, [-1.3, -0.1], [0.15, 0.15]),
                (0.7, [-0.8, -0.4], [0.2, 0.2]),
                (0.8, [-0.8, -0], [0.4, 0.4]),
                (0.4, [0.6, 0.0], [0.4, 0.2]),
                (0.7, [1.25, 0.3], [0.25, 0.25])]
