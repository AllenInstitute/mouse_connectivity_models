"""
Module containing VoxelData and RegionalData objects.
"""
# Authors: Joseph Knox <josephk@alleninstitute.org>
# License: Allen Institute Software License

# TODO: implement __repr__
# TODO: integrate into VoxelModelCache

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
        """Default structure ids."""
        # NOTE: Necessary copy from allensdk.core.MouseConnectivityCache because
        #       of hardcoded class and summary structure set id error due to
        #       new annotation (ccf)

        if not hasattr(self, '_default_structure_ids'):
            tree = self.cache.get_structure_tree()
            default_structures = tree.get_structures_by_set_id(
                self.DEFAULT_STRUCTURE_SET_IDS)
            self._default_structure_ids = [st['id'] for st in default_structures
                                           if st['id'] != 934]

        return self._default_structure_ids

    def __init__(self,
                 cache,
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

        self.cache = cache
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

        self.injection_mask = Mask.from_cache(
            cache,
            structure_ids=self.injection_structure_ids,
            hemisphere_id=self.injection_hemisphere_id)
        self.projection_mask = Mask.from_cache(
            cache,
            structure_ids=self.projection_structure_ids,
            hemisphere_id=self.projection_hemisphere_id)

    def __repr__(self):
        # TODO: update to show parameters
        return "{}()".format(self.__class__.__name__)

    def _experiment_generator(self, experiment_ids):
        """Generates experiment objections given their experiment ids"""
        def valid_volume(experiment):
            """Does experiment meet volume requirements?"""
            # compute injection ratio contained within injection mask
            contained_injection = self.injection_mask.mask_volume(
                experiment.injection_density)
            contained_ratio = contained_injection.sum() / experiment.injection_volume

            # convert to mm^3
            resolution = self.cache.get_reference_space().resolution[0]
            convert = lambda x: x * (1e-3 * resolution)**3

            injection_volume = convert(experiment.injection_volume)
            projection_volume = convert(experiment.projection_volume)

            return (injection_volume > self.injection_volume_bounds[0] and
                    injection_volume < self.injection_volume_bounds[1] and
                    projection_volume > self.projection_volume_bounds[0] and
                    projection_volume < self.projection_volume_bounds[1] and
                    contained_ratio > self.min_contained_injection_ratio)


        for eid in experiment_ids:
            experiment = Experiment.from_cache(
                self.cache, eid, self.data_mask_tolerance)

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
    cache - VoxelModelCache or MouseConnectivityCache object
        Provides way to pull experiment grid-data from Allen Brain Atlas

    injection_structure_ids : list, optional, default None
        List of structure_ids to which the injection mask will be constrained.

    projection_structure_ids : list, optional, default None
        List of structure_ids to which the projection mask will be constrained.

    injection_hemisphere_id : int, optional, default 3
        Hemisphere (1:left, 2:right, 3:both) to which the injection mask will
        be constrained.

    projection_hemisphere_id : int, optional, default 3
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
    >>> from mcmodels.core import VoxelData, VoxelModelCache
    >>> cache = VoxelModelCache()
    >>> experiment_ids = (112514202, 139520203)
    >>> voxel_data = VoxelData(cache)
    >>> voxel_data.get_experiment_data(experiment_ids)
    VoxelData()
    """
    COARSE_STRUCTURE_SET_ID = 2
    DEFAULT_STRUCTURE_SET_IDS = tuple([COARSE_STRUCTURE_SET_ID])

    def get_experiment_data(self, experiment_ids):
        """Pulls voxel-scale grid data for experiments.

        Uses the cache property to pull grid data from the Allen Brain Atlas.
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
        return RegionalData.from_voxel_data(self)


class RegionalData(_BaseData):
    """Container class for regionalized voxel-scale grid data.

    Parameters
    ----------
    cache - VoxelModelCache or MouseConnectivityCache object
        Provides way to pull experiment grid-data from Allen Brain Atlas

    injection_structure_ids : list, optional, default None
        List of structure_ids to which the injection mask will be constrained.

    projection_structure_ids : list, optional, default None
        List of structure_ids to which the projection mask will be constrained.

    injection_hemisphere_id : int, optional, default 3
        Hemisphere (1:left, 2:right, 3:both) to which the injection mask will
        be constrained.

    projection_hemisphere_id : int, optional, default 3
        Hemisphere (1:left, 2:right, 3:both) to which the projection mask will
        be constrained.

    normalized_injection : boolean, optional, default True
        If True, the injection density will be normalized by the total
        injection density for each experiment.

    normalized_projection : boolean, optional, default True
        If True, the projection density will be normalized by the total
        injection density for each experiment.

    flip_experiments : boolean, optional, default True
        If True, experiment grid-data will be reflected across the midline.
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

    Notes
    -----
    - :meth:`RegionalData.from_voxel_data` will not return a :class:`RegionalData`
    object having identical ``injections`` or ``projections`` attributes to those
    generated from :meth:`get_experiment_data` since the latter unionizes are
    computed at a finer resolution. Still, the results should be very similar.
    - :meth:`get_experiment_data` is prefered if only concerned with unionized
    data, because it loads the unionizes cached by the ``cache`` parameter
    instead of computing the unionizations from cached grid data volumes.

    See also
    --------
    VoxelData

    Examples
    --------
    >>> from mcmodels.core import RegionalData, VoxelModelCache
    >>> cache = VoxelModelCache()
    >>> experiment_ids = (112514202, 139520203)
    >>> regional_data = RegionalData(cache)
    >>> regional_data.get_experiment_data(experiment_ids)
    RegionalData()
    """

    ROOT_STRUCTURE_ID = 997
    SUMMARY_STRUCTURE_SET_ID = 687527945
    DEFAULT_STRUCTURE_SET_IDS = tuple([SUMMARY_STRUCTURE_SET_ID])

    @classmethod
    def from_voxel_data(cls, voxel_data):
        """Construct class from a VoxelData object.

        Parameters
        ----------
        voxel_data : a VoxelData object

        Returns
        -------
        RegionalData : an installation of the RegionalData object
        """
        regional = cls(voxel_data.cache,
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

        if hasattr(voxel_data, 'injections'):
            injection_key = voxel_data.injection_mask.get_key(
                structure_ids=voxel_data.injection_structure_ids,
                hemisphere_id=voxel_data.injection_hemisphere_id)
            projection_key = voxel_data.projection_mask.get_key(
                structure_ids=voxel_data.projection_structure_ids,
                hemisphere_id=voxel_data.projection_hemisphere_id)

            regional.injections = unionize(voxel_data.injections, injection_key)
            regional.projections = unionize(voxel_data.projections, projection_key)

        return regional

    def _subset_experiments_by_injection_hemisphere(self, unionizes):
        def _get_hemisphere_injection_map():
            hemisphere_injections = {1 : [], 2 : []}
            for eid in unionizes.experiment_id.unique():
                eid_inj = (unionizes.is_injection) & (unionizes.experiment_id == eid)
                l_sum = unionizes[
                    eid_inj & (unionizes.hemisphere_id == 1)].projection_density.sum()
                r_sum = unionizes[
                    eid_inj & (unionizes.hemisphere_id == 2)].projection_density.sum()

                inj_hemi = 1 if l_sum > r_sum else 2
                hemisphere_injections[inj_hemi].append(eid)

            return hemisphere_injections

        hemi_injection_map = _get_hemisphere_injection_map()

        if self.injection_hemisphere_id in [1, 2]:
            if self.flip_experiments:
                # map 2 -> 1 and map 1 -> 2
                other_hemisphere_id = 3 - self.injection_hemisphere_id
                to_flip = hemi_injection_map[other_hemisphere_id]
                rows = unionizes.experiment_id.isin(to_flip)

                l_rows = rows & (unionizes.hemisphere_id == 1)
                r_rows = rows & (unionizes.hemisphere_id == 2)

                unionizes.loc[l_rows, 'hemisphere_id'] = 2
                unionizes.loc[r_rows, 'hemisphere_id'] = 1
            else:
                valid_eids = hemi_injection_map[self.injection_hemisphere_id]
                unionizes = unionizes[unionizes.experiment_id.isin(valid_eids)]

        return unionizes


    def _subset_experiments_by_volume_parameters(self, unionizes, experiment_ids):
        def valid_experiment(eid):
            exp_unionizes = unionizes[unionizes.experiment_id == eid]
            root = (exp_unionizes.structure_id == self.ROOT_STRUCTURE_ID) &\
                   (exp_unionizes.hemisphere_id == 3)

            injection_volume = exp_unionizes[
                root & exp_unionizes.is_injection].projection_volume.values
            projection_volume = exp_unionizes[
                root & (~exp_unionizes.is_injection)].projection_volume.values

            contained_injection = exp_unionizes[
                exp_unionizes.structure_id.isin(self.injection_structure_ids) &
                (exp_unionizes.hemisphere_id == 3) &
                exp_unionizes.is_injection].projection_volume
            contained_ratio = contained_injection.sum() / injection_volume

            return all((injection_volume >= self.injection_volume_bounds[0],
                        injection_volume <= self.injection_volume_bounds[1],
                        projection_volume >= self.projection_volume_bounds[0],
                        projection_volume <= self.projection_volume_bounds[1],
                        contained_ratio >= self.min_contained_injection_ratio))

        valid_eids = [eid for eid in experiment_ids if valid_experiment(eid)]

        return unionizes[unionizes.experiment_id.isin(valid_eids)]


    def get_experiment_data(self, experiment_ids, use_dataframes=False):
        """Pulls regionalized voxel-scale grid data for experiments.

        Uses the cache attribute to pull grid data from the Allen Brain Atlas.
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
        def _get_data_array(unionizes, is_injection, normalized, hemisphere_id):
            index = 'experiment_id'
            columns = 'structure_id'
            values = 'normalized_projection_volume' if normalized else 'projection_density'

            valid_rows = (unionizes.is_injection == is_injection) &\
                         (unionizes.hemisphere_id == hemisphere_id)

            # better imo than using pd.pivot_table for safety's sake
            return unionizes[valid_rows].pivot(index=index, columns=columns, values=values)

        all_structure_ids = list(set([self.ROOT_STRUCTURE_ID]) |
                                 set(self.injection_structure_ids) |
                                 set(self.projection_structure_ids))
        unionizes = self.cache.get_structure_unionizes(
            experiment_ids, structure_ids=all_structure_ids)

        # subset unionize rows
        unionizes = self._subset_experiments_by_volume_parameters(unionizes, experiment_ids)
        unionizes = self._subset_experiments_by_injection_hemisphere(unionizes)


        injections = _get_data_array(
            unionizes, True, self.normalized_injection, self.injection_hemisphere_id)

        projections = _get_data_array(
            unionizes, False, self.normalized_projection, self.projection_hemisphere_id)

        # fill empty injection structures
        missing = set(self.injection_structure_ids) - set(injections.columns.values)
        for sid in missing:
            injections[sid] = 0.

        # projections is found using is_injection=False, add back injection
        injections = injections.fillna(value=0.0)
        projections = projections.add(injections).fillna(value=0.0)

        # subset to structure ids
        injections = injections[self.injection_structure_ids]
        projections = projections[self.projection_structure_ids]

        if use_dataframes:
            self.injections = injections
            self.projections = projections
        else:
            self.injections = injections.values
            self.projections = projections.values

        return self
