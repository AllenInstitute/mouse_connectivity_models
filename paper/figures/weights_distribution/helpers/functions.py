from abc import ABCMeta, abstractmethod, abstractproperty

import six
import numpy as np

def _rsquared(x, y):
    """from scipy.stats.stats.linregress"""
    ssxm, ssxym, ssyxm, ssym = np.cov(x, y, bias=1).flat
    r_num = ssxym
    r_den = np.sqrt(ssxm*ssym)
    if r_den == 0.0:
        return 0.0
    r = r_num / r_den
    # test for numerical error propogation
    if (r > 1.0):
        return 1.0
    elif (r < -1.0):
        return -1.0

    return r**2


class _BaseResiduals(six.with_metaclass(ABCMeta)):

    def __init__(self, x0=None):
        self.x0 = x0
        if x0 is None:
            self.x0 = np.ones(self.n_coeffs)
        elif isinstance(x0, np.ndarray) and x0.ndim != 1:
            raise NotImplementedError("Please pass x0 as a 1D array")

    @abstractproperty
    def n_coeffs(self):
        """n_coeffs"""

    @abstractmethod
    def predict(self, t, y):
        """prediction function"""

    def rsquared(self, x, t, y):
        return _rsquared(self.predict(x, t), np.log(y))


    def __call__(self, x, t, y):
        """calls"""
        return self.predict(x, t) - np.log(y)


class LogLog(_BaseResiduals):

    n_coeffs = 2

    def predict(self, x, t):
        if np.any(t <= 0):
            raise ValueError("t must be nonnegative")
        return x[0] + x[1]*np.log(t)


class LogLinear(_BaseResiduals):

    n_coeffs = 2

    def predict(self, x, t):
        if np.any(t <= 0):
            raise ValueError("t must be nonnegative")
        return x[0] + x[1]*t
