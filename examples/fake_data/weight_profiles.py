"""
weight profiles for test case
"""
import numpy as np

def ridge_and_bump(size,
                   ridge_sigma=0.3,
                   bump_sigma=0.12,
                   bump_scale=0.9,
                   bump_xloc=0.8,
                   bump_yloc=0.15):
    """ computes weights as ... """
    a = np.linspace(0, 1, size)
    x, y = np.meshgrid(*[a,a])

    # ridge
    ridge = np.exp((-(x - y / ridge_sigma) ** 2))

    # gaussian bump
    num = (x - bump_xloc) - (y - bump_yloc)
    bump = bump_scale*np.exp((num / bump_sigma)**2)

    return ridge + bump

def mexican_hat(size):
    raise NotImplementedError

def random_gaussians(size):
    raise NotImplementedError
