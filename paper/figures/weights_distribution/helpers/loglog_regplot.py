import copy

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from seaborn import algorithms as algo
from seaborn.regression import _RegressionPlotter

class _HackPlotter(_RegressionPlotter):
    """Hacky way to get loglog regplots"""

    def fit_logx(self, grid):
        """Fit the model in log-space."""
        X, y = np.c_[np.ones(len(self.x)), self.x], self.y
        grid = np.c_[np.ones(len(grid)), np.log(grid)]

        def reg_func(_x, _y):
            _x = np.c_[_x[:, 0], np.log(_x[:, 1])]
            _y = np.log(_y)

            return np.linalg.pinv(_x).dot(_y)

        self.betas = reg_func(X, y)
        yhat = grid.dot(self.betas)
        if self.ci is None:
            return np.exp(yhat), None

        beta_boots = algo.bootstrap(X, y, func=reg_func,
                                    n_boot=self.n_boot, units=self.units).T
        yhat_boots = grid.dot(beta_boots).T

        return np.exp(yhat), np.exp(yhat_boots)


def loglog_regplot(*args, **kwargs):
    """hacky way to get loglog plot in seaborn"""

    # pull off
    ax = kwargs.pop('ax', None)
    marker = kwargs.pop('marker', 'o')
    scatter_kws = kwargs.pop('scatter_kws', None)
    line_kws = kwargs.pop('line_kws', None)

    plotter_kws = dict(logx=True, lowess=False, robust=False, logistic=False)

    kwargs.update(plotter_kws)
    plotter = _HackPlotter(*args, **kwargs)

    if ax is None:
        ax = plt.gca()

    scatter_kws = {} if scatter_kws is None else copy.copy(scatter_kws)
    scatter_kws["marker"] = marker
    line_kws = {} if line_kws is None else copy.copy(line_kws)
    plotter.plot(ax, scatter_kws, line_kws)
    return ax, plotter.betas
