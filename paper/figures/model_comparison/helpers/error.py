import abc
import six

import numpy as np
from sklearn.model_selection import GridSearchCV, LeaveOneOut, cross_validate
from mcmodels.models import HomogeneousModel, VoxelModel
from mcmodels.regressors import NadarayaWatson, NadarayaWatsonCV

from .params import RBFParams, PolynomialParams
from .scorers import HybridScorer, mse_rel, regional_mse_rel


class NestedCV(object):
    def __init__(self, cv=None, scoring=None, n_jobs=-1, return_train_score=True):
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.return_train_score = return_train_score

    def fit(self, estimator, X, y, param_grid):
        reg = GridSearchCV(estimator=estimator, cv=self.cv,
                           param_grid=param_grid, scoring=self.scoring)

        self.scores = cross_validate(reg, X, y, cv=self.cv, n_jobs=self.n_jobs,
                                     return_train_score=self.return_train_score,
                                     scoring=self.scoring)

        return self


class _BaseError(six.with_metaclass(abc.ABCMeta)):

    @abc.abstractmethod
    def run(self):
        '''run cv'''


class VoxelModelError(_BaseError):

    LOG_CONSTANT = 1e-8
    DEFAULT_KERNEL = 'rbf'
    DEFAULT_CV = LeaveOneOut()

    def __init__(self, cache, voxel_data, structure_ids=None, kernel=None,
                 kernel_params=dict(), cv=None, return_train_score=True, n_jobs=-1):
        if kernel is None:
            self.kernel = self.DEFAULT_KERNEL
        if cv is None:
            self.cv = self.DEFAULT_CV

        self.cache = cache
        self.voxel_data = voxel_data
        self.structure_ids = structure_ids
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.cv = cv
        self.return_train_score = return_train_score
        self.n_jobs = n_jobs

    @property
    def scoring_dict(self):
        try:
            return self._scoring_dict
        except AttributeError:
            scorer = HybridScorer(self.cache, structure_ids=self.structure_ids)
            self._scoring_dict = scorer.scoring_dict
            return self._scoring_dict

    @property
    def param_grid(self):
        if self.kernel.lower() == 'rbf':
            params = RBFParams(**self.kernel_params)
        elif self.kernel.lower() == 'polynomial':
            params = PolynomialParams(self.voxel_data, **self.kernel_params)

        return params.param_grid

    def single_cv(self, option='standard'):
        y = self.voxel_data.projections
        if option.lower() in ('log', 'logged'):
            y = np.log10(y + self.LOG_CONSTANT)

        estimator = NadarayaWatsonCV(self.param_grid, scoring=self.scoring_dict['voxel'])
        estimator.fit(self.voxel_data.centroids, y)

        return estimator

    def fit(self, option='standard', **kwargs):
        y = self.voxel_data.projections
        if option.lower() in ('log', 'logged'):
            y = np.log10(y + self.LOG_CONSTANT)

        estimator = NadarayaWatson(**kwargs)
        estimator.fit(self.voxel_data.centroids, y)

        return estimator

    def _run_nadaraya_watson_cv(self, y):
        '''helper: easy transform of data'''
        estimator = NadarayaWatsonCV(self.param_grid, scoring=self.scoring_dict['voxel'])
        self.scores = cross_validate(estimator,
                                     X=self.voxel_data.centroids,
                                     y=y,
                                     cv=self.cv,
                                     scoring=self.scoring_dict,
                                     return_train_score=self.return_train_score,
                                     n_jobs=self.n_jobs)
        return self

    def run_standard_model_error(self):
        self._run_nadaraya_watson_cv(self.voxel_data.projections)

    def run_log_model_error(self):
        log = lambda y: np.log10(y + self.LOG_CONSTANT)
        self._run_nadaraya_watson_cv(log(self.voxel_data.projections))

    def run_injection_model_error(self):
        estimator = VoxelModel(self.voxel_data.injection_mask.coordinates)
        X = np.hstack((self.voxel_data.centroids, self.voxel_data.injections))
        y = self.voxel_data.projections

        # TODO: work in above
        reg = GridSearchCV(estimator=estimator, cv=self.cv,
                           param_grid=self.param_grid,
                           scoring=self.scoring_dict['voxel'])

        self.scores = cross_validate(reg, X, y, cv=self.cv, n_jobs=self.n_jobs,
                                     return_train_score=self.return_train_score,
                                     scoring=self.scoring_dict)

        return self

    def run(self, option='standard'):
        if option.lower() == 'standard':
            self.run_standard_model_error()

        elif option.lower() in ('log', 'logged'):
            self.run_log_model_error()

        elif option.lower() in ('inj', 'injection', 'injection-based'):
            self.run_injection_model_error()

        else:
            raise ValueError('option %s is not valid' % option)


class HomogeneousModelError(object):

    def __init__(self, cache, regional_data, cv=None, return_train_score=True, n_jobs=-1):
        self.cache = cache
        self.regional_data = regional_data
        self.cv = cv
        self.return_train_score = return_train_score
        self.n_jobs = n_jobs

    def run(self):
        estimator = HomogeneousModel(kappa=np.inf)
        param_grid = dict(kappa=[np.inf])
        X = self.regional_data.injections
        y = self.regional_data.projections

        cv = NestedCV(cv=self.cv, scoring=mse_rel(), n_jobs=self.n_jobs,
                       return_train_score=self.return_train_score)
        cv.fit(estimator, X, y, param_grid)

        self.scores = cv.scores
        self.scores['test_regional'] = self.scores.pop('test_score')
        if self.return_train_score:
            self.scores['train_regional'] = self.scores.pop('train_score')

        return self
