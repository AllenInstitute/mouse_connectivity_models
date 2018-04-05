from __future__ import division, print_function
from collections import namedtuple

import numpy as np
import scipy.stats as stats
import scipy.optimize as sopt


def _rmse(fun):
    return np.sqrt(np.mean(fun**2))

def _eval_model_fit(f, args, kwargs, **ltsq_kwargs):
    res = sopt.least_squares(f, f.x0, args=args, kwargs=kwargs, **ltsq_kwargs)
    rmse = _rmse(res.fun)
    rsquared = f.rsquared(*args)

    return res.x, res.cost, res.fun, res.grad, res.optimality, rmse, rsquared


class LtsqFitResult(namedtuple('LtsqFitResult',
                               ('name', 'f', 'x', 'cost', 'fun', 'grad',
                                'optimality', 'rmse', 'rsquared'))):
    """A DistFit fit results"""
    __slots__ = ()

    def __new__(cls, name, f, x, cost, fun, grad, optimality, rmse, rsquared):
        return super(LtsqFitResult, cls).__new__(
            cls, name, f, x, cost, fun, grad, optimality, rmse, rsquared)

    def __eq__(self, other):
        return self.cost == other.cost

    def __lt__(self, other):
        return self.cost < other.cost

    def __le__(self, other):
        return self.cost <= other.cost


class LtsqFit(object):

    def __init__(self, functions, **least_squares_kwargs):
        self.functions = functions
        self.least_squares_kwargs = least_squares_kwargs

    def fit(self, *args, **kwargs):

        results = []
        for f in self.functions:
            name = f.__class__.__name__
            fit = _eval_model_fit(f, args, kwargs, **self.least_squares_kwargs)

            results.append(LtsqFitResult(name, f, *fit))

        self.results_ = results
        self.best_fit_ = min(results)
        self.best_f_ = self.best_fit_.f

        return self

    def predict(self, args=()):
        """get pdf @ x of best"""
        if not hasattr(self, "results_"):
            raise ValueError("must first fit!!")

        return self.best_fit_.f.predict(self.best_fit_.x, *args)

    def _results_table(self):
        """pretty printing of results"""
        def print_row(result, keys):
            row = []
            for k in keys:
                r = getattr(result, k)
                if not isinstance(r, str):
                    r = "{:.3}".format(r)
                row.append(r)

            return row, max(map(len, row))

        def join_row(row, maxlen):
            l = row[0].ljust(maxlen)
            r = "".join(x.rjust(maxlen) for x in row[1:])

            return l + r + "\n"

        header = "Fit Results"
        col_labels = ("name", "rmse", "cost", "optimality", "rsquared")
        col_underlines = ["-"*len(x) for x in col_labels]

        # break up into rows
        rows = [print_row(r, col_labels) for r in sorted(self.results_)]
        rows, maxes = zip(*rows)

        maxlen = max(maxes) + 2 # padding
        width = maxlen*len(col_labels)

        # concatenate return string
        s = '\n{:^{width}}\n'.format(header, width=width)
        s += width*"=" + "\n"
        s += join_row(col_labels, maxlen)
        s += join_row(col_underlines, maxlen)
        s += "".join(join_row(r, maxlen) for r in rows)
        return s

    def print_results(self):
        """pretty printing of results"""
        if not hasattr(self, "results_"):
            raise ValueError("must first fit!!")
        s = self._results_table()
        print(s)

    def to_dict(self):
        """returns dict of results"""
        if not hasattr(self, "results_"):
            raise ValueError("must first fit!!")

        d = dict()
        for res in self.results_:
            res_dict = res._asdict()
            name = res_dict.pop('name')
            d[name] = dict(res_dict)

        return d

    def __repr__(self):
        return "{0}(functions={1})".format(
            self.__class__.__name__,
            (f.__class__.__name__ for f in self.functions))

    def __str__(self):
        if not hasattr(self, "results_"):
            return self.__repr__()
        return self._results_table()
