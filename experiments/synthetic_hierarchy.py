"""
Synthetic experiment to study hierarchical feature emergence
in simple deep networks.
"""

import numpy as np

def generate_hierarchical_data(n=1000):
    x = np.random.randn(n, 2)
    y = (x[:, 0] * x[:, 1] > 0).astype(int)
    return x, y
