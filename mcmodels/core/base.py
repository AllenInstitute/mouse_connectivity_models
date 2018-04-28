"""
Module containing VoxelData and RegionalData objects.
"""
# Authors: Joseph Knox <josephk@alleninstitute.org>
# License: Allen Institute Software License

from abc import ABCMeta, abstractmethod, abstractproperty

import six
import numpy as np

from .experiment import Experiment
from .masks import Mask
from ..utils import unionize


class _BaseData(six.with_metaclass(ABCMeta)):

    @abstractproperty
    def DEFAULT_STRUCTURE_SET_IDS(self):
        """ for default structure ids """

    @property
    def default_structure_ids(self):
        """Returns default structure ids.

        ..note:
        taken from allensdk.core.mouse_connectivity_cache.VoxelModelCache
        """
        if not hasattr(self, '_default_structure_ids'):
            tree = self.voxel_model_cache.get_structure_tree()
            default_structures = tree.get_structures_by_set_id(
                self.DEFAULT_STRUCTURE_SET_IDS)
            self._default_structure_ids = [st['id']
                                           for st in default_structures]

        return self._default_structure_ids

    def __init__(self,
                 voxel_model_cache,
                 injection_structure_ids=None,
                 projection_structure_ids=None,
                 injection_hemisphere_id=3,
                 projection_hemisphere_id=3,
                 normalized_injection=True,
                 normalized_projection=True,
                 flip_experiments=True,
                 data_mask_tolerance=0.0,
                 injection_volume_bounds=(0.0, np.inf),
                 projection_volume_bounds=(0.0, np.inf),
                 min_contained_injection_ratio=0.0):

        self.voxel_model_cache = voxel_model_cache
        self.injection_structure_ids = injection_structure_ids
        self.projection_structure_ids = projection_structure_ids
        self.injection_hemisphere_id = injection_hemisphere_id
        self.projection_hemisphere_id = projection_hemisphere_id
        self.normalized_injection = normalized_injection
        self.normalized_projection = normalized_projection
        self.flip_experiments = flip_experiments
        self.data_mask_tolerance = data_mask_tolerance
        self.injection_volume_bounds = injection_volume_bounds
        self.projection_volume_bounds = projection_volume_bounds
        self.min_contained_injection_ratio = min_contained_injection_ratio

        if self.injection_structure_ids is None:
            self.injection_structure_ids = self.default_structure_ids

        if self.projection_structure_ids is None:
            self.projection_structure_ids = self.default_structure_ids

        self.injection_mask = Mask(voxel_model_cache=self.voxel_model_cache,
                                   structure_ids=self.injection_structure_ids,
                                   hemisphere_id=self.injection_hemisphere_id)
        self.projection_mask = Mask(voxel_model_cache=self.voxel_model_cache,
                                    structure_ids=self.projection_structure_ids,
                                    hemisphere_id=self.projection_hemisphere_id)

    def _experiment_generator(self, experiment_ids):
        """Generates experiment objections given their experiment ids"""
        def valid_volume(experiment):
            """Does experiment meet volume requirements?"""
            # compute injection ratio contained within injection mask
            contained_injection = self.injection_mask.mask_volume(
                experiment.injection_density)
            contained_ratio = contained_injection.sum() / experiment.injection_volume

            # convert to mm^3
            resolution = self.voxel_model_cache.get_reference_space().resolution[0]
            convert = lambda x: x * (1e-3 * resolution)**3

            injection_volume = convert(experiment.injection_volume)
            projection_volume = convert(experiment.projection_volume)

            return (injection_volume > self.injection_volume_bounds[0] and
                    injection_volume < self.injection_volume_bounds[1] and
                    projection_volume > self.projection_volume_bounds[0] and
                    projection_volume < self.projection_volume_bounds[1] and
                    contained_ratio > self.min_contained_injection_ratio)


        for eid in experiment_ids:
            experiment = Experiment.from_voxel_model_cache(
                self.voxel_model_cache, eid, self.data_mask_tolerance)

            if valid_volume(experiment):
                hemisphere_id = experiment.injection_hemisphere_id

                if (self.injection_hemisphere_id == 3 or
                        hemisphere_id == self.injection_hemisphere_id):
                    yield experiment

                elif self.flip_experiments:
                    yield experiment.flip()

    @abstractmethod
    def get_experiment_data(self, experiment_ids):
        """forms data arrays, returns self"""


class VoxelData(_BaseData):
    """Container class for voxel-scale grid data.

    Parameters
    ----------
    voxel_model_cache - VoxelModelCache object
        Provides way to pull experiment grid-data from Allen Brain Atlas

    injection_structure_ids : list, optional, default None
        List of structure_ids to which the injection mask will be constrained.

    projection_structure_ids : list, optional, default None
        List of structure_ids to which the projection mask will be constrained.

    injection_hemisphere_id : int, optional, defualt 3
        Hemisphere (1:left, 2:right, 3:both) to which the injection mask will
        be constrained.

    projection_hemisphere_id : int, optional, defualt 3
        Hemisphere (1:left, 2:right, 3:both) to which the projection mask will
        be constrained.

    normalized_injection : boolean, optional, default True
        If True, the injection density will be normalized by the total
        injection density for each experiment.

    normalized_projection : boolean, optional, default True
        If True, the projection density will be normalized by the total
        injection density for each experiment.

    flip_experiments : boolean, optional, default True
        If True, experiment grid-data will be refelcted accross the midline.
        Useful if you wish to include L hemisphere injections into a R
        hemisphere model.

    data_mask_tolerance : float, optional, default 0.0
        Tolerance with which to include data in voxels informatically labeled
        as having error. The data_mask for each experiment is an array with
        values between (0, 1), where 1 indicates the voxel fully contains an
        error, whereas 0 indicates the voxel does not contain any error. A value
        of 0.0 thus indicates the highest threshold for data, whereas a value of
        1.0 indicates that data will be included from all voxels.

    injection_volume_bounds : float, optional, default (0.0, np.inf)
        Includes experiments with total injection volume (mm^3) within bounds.

    projection_volume_bounds : float, optional, default (0.0, np.inf)
        Includes experiments with total projection volume (mm^3) within bounds.

    min_contained_injection_ratio : float, optional, default 0.0
        Includes experiments with total injection volume ratio within injection
        mask.

    Attributes
    ----------
    injection_mask : Mask object
        Mask object used to constrain and flatten the injection_density from
        each experiment. This object can also be used to generate a key relating
        each column of the injections matrix to a corresponding structure or to
        transform a given row of the injections matrix to its corresponding
        brain volume.

    projection_mask : Mask object
        Mask object used to constrain and flatten the projection_density from
        each experiment. This object can also be used to generate a key relating
        each column of the projections matrix to a corresponding structure or to
        transform a given row of the projections matrix to its corresponding
        brain volume.

    centroids : array, shape (n_experiments, 3)
        Stacked array of injection centroids for each experiment.

    injections : array, shape (n_experiments, n_source_voxels)
        Stacked array of constrained, flattened injection densities for each
        experiment.

    projections : array, shape (n_experiments, n_target_voxels)
        Stacked array of constrained, flattened projection densities for each
        experiment.

    See also
    --------
    RegionalData

    Examples
    --------
    >>> from voxel_model import VoxelData, VoxelModelCache
    >>> voxel_model_cache = VoxelModelCache()
    >>> experiment_ids = (112514202, 139520203)
    >>> voxel_data = VoxelData()
    >>> voxel_data.get_experiment_data(experiment_ids)
    xx
    """
    DEFAULT_STRUCTURE_SET_ID = 2
    DEFAULT_STRUCTURE_SET_IDS = tuple([DEFAULT_STRUCTURE_SET_ID])

    def get_experiment_data(self, experiment_ids):
        """Pulls voxel-scale grid data for experiments.

        Uses the voxel_model_cache property to pull grid data from the Allen Brain Atlas.
        Note that only experiments passing all defined parameters will be
        included.

        Parameters
        ----------
        experiment_ids : list
            Ids of candidate experiments to pull. Only the subset of these
            experiments passing user defined object parameters will be pulled.

        Returns
        -------
        self : returns an instance of self.
        """
        def get_centroid(experiment):
            """Returns experiment centroid"""
            return experiment.centroid

        def get_injection(experiment):
            """Returns experiment injection masked & flattened"""
            injection = experiment.get_injection(self.normalized_injection)
            return self.injection_mask.mask_volume(injection)

        def get_projection(experiment):
            """Returns experiment projection masked & flattened"""
            projection = experiment.get_projection(self.normalized_projection)
            return self.projection_mask.mask_volume(projection)

        # get data
        get_data = lambda x: (get_centroid(x),
                              get_injection(x),
                              get_projection(x))
        arrays = map(get_data, self._experiment_generator(experiment_ids))

        centroids, injections, projections = map(np.array, zip(*arrays))
        self.centroids = centroids
        self.injections = injections
        self.projections = projections

        return self

    def get_regional_data(self):
        """Returns RegionalData object with same parameters."""
        regional_data = RegionalData.from_voxel_data(self)

        if hasattr(self, 'centroids'):
            # unionize pulled data
            regional_data.centroids = self.centroids.copy()
            regional_data.injections = self.injections.copy()
            regional_data.projections = self.projections.copy()

            # NOTE: a little hacky :: accessing private method outside class
            regional_data._unionize_experiment_data()

        return regional_data


class RegionalData(_BaseData):
    """Container class for regionalized voxel-scale grid data.

    Parameters
    ----------
    voxel_model_cache - VoxelModelCache object
        Provides way to pull experiment grid-data from Allen Brain Atlas

    injection_structure_ids : list, optional, default None
        List of structure_ids to which the injection mask will be constrained.

    projection_structure_ids : list, optional, default None
        List of structure_ids to which the projection mask will be constrained.

    injection_hemisphere_id : int, optional, defualt 3
        Hemisphere (1:left, 2:right, 3:both) to which the injection mask will
        be constrained.

    projection_hemisphere_id : int, optional, defualt 3
        Hemisphere (1:left, 2:right, 3:both) to which the projection mask will
        be constrained.

    normalized_injection : boolean, optional, default True
        If True, the injection density will be normalized by the total
        injection density for each experiment.

    normalized_projection : boolean, optional, default True
        If True, the projection density will be normalized by the total
        injection density for each experiment.

    flip_experiments : boolean, optional, default True
        If True, experiment grid-data will be refelcted accross the midline.
        Useful if you wish to include L hemisphere injections into a R
        hemisphere model.

    data_mask_tolerance : float, optional, default 0.0
        Tolerance with which to include data in voxels informatically labeled
        as having error. The data_mask for each experiment is an array with
        values between (0, 1), where 1 indicates the voxel fully contains an
        error, whereas 0 indicates the voxel does not contain any error. A value
        of 0.0 thus indicates the highest threshold for data, whereas a value of
        1.0 indicates that data will be included from all voxels.

    injection_volume_bounds : float, optional, default (0.0, np.inf)
        Includes experiments with total injection volume (mm^3) within bounds.

    projection_volume_bounds : float, optional, default (0.0, np.inf)
        Includes experiments with total projection volume (mm^3) within bounds.

    min_contained_injection_ratio : float, optional, default 0.0
        Includes experiments with total injection volume ratio within injection
        mask.

    Attributes
    ----------
    injection_mask : Mask object
        Mask object used to constrain and flatten the injection_density from
        each experiment. This object can also be used to generate a key relating
        each column of the injections matrix to a corresponding structure or to
        transform a given row of the injections matrix to its corresponding
        brain volume.

    projection_mask : Mask object
        Mask object used to constrain and flatten the projection_density from
        each experiment. This object can also be used to generate a key relating
        each column of the projections matrix to a corresponding structure or to
        transform a given row of the projections matrix to its corresponding
        brain volume.

    centroids : array, shape (n_experiments, 3)
        Stacked array of injection centroids for each experiment.

    injections : array, shape (n_experiments, n_source_regions)
        Stacked array of constrained, flattened injection densities for each
        experiment.

    projections : array, shape (n_experiments, n_target_regions)
        Stacked array of constrained, flattened projection densities for each
        experiment.

    See also
    --------
    VoxelData

    Examples
    --------
    >>> from voxel_model import RegionalData, VoxelModelCache
    >>> voxel_model_cache = VoxelModelCache()
    >>> experiment_ids = (112514202, 139520203)
    >>> voxel_data = RegionalData()
    >>> voxel_data.get_experiment_data(experiment_ids)
    xx
    """

    DEFAULT_STRUCTURE_SET_ID = 167587189
    DEFAULT_STRUCTURE_SET_IDS = tuple([DEFAULT_STRUCTURE_SET_ID])

    @classmethod
    def from_voxel_data(cls, voxel_data):
        """Construct class from a VoxelData object.

        Parameters
        ----------
        voxel_data : a VoxelData object

        Returns
        -------
        RegionalData : an instatiation of the RegionalData object
        """
        return cls(voxel_data.voxel_model_cache,
                   injection_structure_ids=voxel_data.injection_structure_ids,
                   projection_structure_ids=voxel_data.projection_structure_ids,
                   injection_hemisphere_id=voxel_data.injection_hemisphere_id,
                   projection_hemisphere_id=voxel_data.projection_hemisphere_id,
                   normalized_injection=voxel_data.normalized_injection,
                   normalized_projection=voxel_data.normalized_projection,
                   flip_experiments=voxel_data.flip_experiments,
                   data_mask_tolerance=voxel_data.data_mask_tolerance,
                   injection_volume_bounds=voxel_data.injection_volume_bounds,
                   projection_volume_bounds=voxel_data.projection_volume_bounds,
                   min_contained_injection_ratio=voxel_data.min_contained_injection_ratio)

    def _unionize_experiment_data(self):
        """Private helper method to unionize voxel scale data to regions."""
        injection_key = self.injection_mask.get_key()
        projection_key = self.projection_mask.get_key()

        self.injections = unionize(self.injections, injection_key)
        self.projections = unionize(self.projections, projection_key)

        return self

    def get_experiment_data(self, experiment_ids):
        """Pulls regionalized voxel-scale grid data for experiments.

        Uses the voxel_model_cache attribute to pull grid data from the Allen Brain Atlas.
        Note that only experiments passing all defined parameters will be
        included.

        Parameters
        ----------
        experiment_ids : list
            Ids of candidate experiments to pull. Only the subset of these
            experiments passing user defined object parameters will be pulled.

        Returns
        -------
        self : returns an instance of self.
        """
        super(RegionalData, self).get_experiment_data(experiment_ids)

        return self._unionize_experiment_data()
