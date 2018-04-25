# -*- coding: utf-8 -*-

# Authors: Joseph Knox <josephk@alleninstitute.org.
# License:

from __future__ import division, print_function

from collections import namedtuple

import numpy as np
import scipy.stats as stats


def _aic(nnlf, k):
    """Akaike information criterion

        2*k - 2*log(Likelihood)
    """
    return 2*(k + nnlf)


def _bic(nnlf, k, n):
    """Bayesian information criterion

        log(n)*k - 2*log(Likelihood)
    """
    # -2*L + k*(np.log(n)-np.log(2*np.pi))
    return np.log(n)*k + 2*nnlf


def _eval_model_fit(x, distname):
    """fits model"""
    try:
        dist = getattr(stats, distname)
    except AttributeError as e:
        raise ValueError("provided dist not in scipy.stats:\n\n%s" % e)

    # fit max like ests and get negative loglikelihood function
    mles = dist.fit(x)
    nnlf = dist.nnlf(mles, x)

    # information criteria
    n, k = map(len, (x, mles))
    aic = _aic(nnlf, k)
    bic = _bic(nnlf, k, n)

    # kolmogrov-s someone test for significance
    ks_stat, p_value = stats.kstest(x, distname, args=mles)

    return DistFitResult(distname, mles, aic, bic, ks_stat, p_value)


class DistFitResult(namedtuple('DistFitResult',
                               ('name', 'mles', 'aic', 'bic',
                                'ks_stat', 'p_value'))):
    """A DistFit fit results"""
    __slots__ = ()

    def __new__(cls, name, mles, aic, bic, ks_stat, p_value):
        return super(DistFitResult, cls).__new__(
            cls, name, mles, aic, bic, ks_stat, p_value)

    def __eq__(self, other):
        return self.bic == other.bic

    def __lt__(self, other):
        return self.bic < other.bic

    def __le__(self, other):
        return self.bic <= other.bic


class DistFit(object):

    def __init__(self, dists):
        self.dists = dists

    def fit(self, x):
        if x.ndim != 1:
            raise NotImplementedError("No current way to compare information "
                                      "criteria for multivariate x")
        results = []
        for name in self.dists:
            results.append(_eval_model_fit(x, name))

        self.results_ = results
        self.best_dist_ = min(results)

        return self

    def predict(self, x):
        """get pdf @ x of best"""
        if not hasattr(self, "results_"):
            raise ValueError("must first fit!!")

        name = self.best_dist_.name
        mles = self.best_dist_.mles

        return getattr(stats, name).pdf(x, *mles)

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

        header = "DistFit Results"
        col_labels = ("name", "aic", "bic", "ks_stat", "p_value")
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
        return "{0}(dists={1})".format(
            self.__class__.__name__, self.dists)

    def __str__(self):
        if not hasattr(self, "results_"):
            return self.__repr__()
        return self._results_table()
