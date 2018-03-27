from abc import ABCMeta, abstractmethod, abstractproperty
from functools import partial

import six
import numpy as np

from .experiment import Experiment
from .masks import Mask


class _BaseModelData(six.with_metaclass(ABCMeta)):

    @abstractproperty
    def DEFAULT_STRUCTURE_SET_IDS(self):
        """ for default structure ids """

    @property
    def default_structure_ids(self):
        """ taken from allenskd """
        if not hasattr(self, '_default_structure_ids'):
            tree = self.mcc.get_structure_tree()
            default_structures = tree.get_structures_by_set_id(self.DEFAULT_STRUCTURE_SET_IDS)
            self._default_structure_ids = [st['id'] for st in default_structures]

        return self._default_structure_ids

    def __init__(self, mcc,
                 injection_structure_ids=None, injection_hemisphere_id=3,
                 projection_structure_ids=None, projection_hemisphere_id=3,
                 normalized_injection=True, normalized_projection=True,
                 flip_experiments=True, data_mask_tolerance=0.0,
                 min_injection_sum=0.0, min_projection_sum=0.0):

        self.mcc = mcc
        self.injection_structure_ids = injection_structure_ids
        self.injection_hemisphere_id = injection_hemisphere_id
        self.projection_structure_ids = projection_structure_ids
        self.projection_hemisphere_id = projection_hemisphere_id
        self.normalized_injection = normalized_injection
        self.normalized_projection = normalized_projection
        self.flip_experiments = flip_experiments
        self.data_mask_tolerance = data_mask_tolerance
        self.min_injection_sum = min_injection_sum
        self.min_projection_sum = min_projection_sum

    def _experiment_generator(self, experiment_ids):

        for eid in experiment_ids:
            experiment = Experiment.from_mcc(self.mcc, eid, self.data_mask_tolerance)

            if (experiment.injection_density.sum() >= self.min_injection_sum and
                    experiment.projection_density.sum() >= self.min_projection_sum):

                hemisphere_id = experiment.injection_hemisphere_id
                if (self.injection_hemisphere_id == 3 or
                        hemisphere_id == self.injection_hemisphere_id):
                    yield experiment

                elif self.flip_experiments:
                    yield experiment.flip()

    @abstractmethod
    def get_experiment_data(self, experiment_ids):
        """forms data arrays, returns self"""


class VoxelModelData(_BaseModelData):

    DEFAULT_STRUCTURE_SET_ID = 2
    DEFAULT_STRUCTURE_SET_IDS = tuple([DEFAULT_STRUCTURE_SET_ID])

    def __init__(self, mcc,
                 injection_structure_ids=None, injection_hemisphere_id=3,
                 projection_structure_ids=None, projection_hemisphere_id=3,
                 normalized_injection=True, normalized_projection=True,
                 flip_experiments=True, data_mask_tolerance=0.0,
                 min_injection_sum=0.0, min_projection_sum=0.0):

        super(VoxelModelData, self).__init__(
            mcc,
            injection_structure_ids=injection_structure_ids,
            injection_hemisphere_id=injection_hemisphere_id,
            projection_structure_ids=projection_structure_ids,
            projection_hemisphere_id=projection_hemisphere_id,
            normalized_injection=normalized_injection,
            normalized_projection=normalized_projection,
            flip_experiments=flip_experiments,
            data_mask_tolerance=data_mask_tolerance,
            min_injection_sum=min_injection_sum,
            min_projection_sum=min_projection_sum)

        if self.injection_structure_ids is None:
            self.injection_structure_ids = self.default_structure_ids

        if self.projection_structure_ids is None:
            self.projection_structure_ids = self.default_structure_ids

        self.injection_mask = Mask(mcc=self.mcc,
                                   structure_ids=injection_structure_ids,
                                   hemisphere=injection_hemisphere_id)
        self.projection_mask = Mask(mcc=self.mcc,
                                    structure_ids=projection_structure_ids,
                                    hemisphere=projection_hemisphere_id)

    def get_experiment_data(self, experiment_ids):
        """forms data arrays, returns self"""
        def get_centroid(experiment):
            return experiment.centroid

        def get_injection(experiment):
            injection = experiment.get_injection(self.normalized_injection)
            return self.injection_mask.mask_volume(injection)

        def get_projection(experiment):
            projection = experiment.get_projection(self.normalized_projection)
            return self.projection_mask.mask_volume(projection)

        # get data
        get_data = lambda x: (get_centroid(x), get_injection(x), get_projection(x))
        arrays = map(get_data, self._experiment_generator(experiment_ids))

        centroids, injections, projections = map(np.array, zip(*arrays))
        self.centroids = centroids
        self.injections = injections
        self.projections = projections

        return self
