"""
Module containing Experiment object and supporting functions.
"""
# Authors: Joseph Knox <josephk@alleninstitute.org>
# License: Allen Institute Software License

from __future__ import division
from functools import partial

import numpy as np

from .utils import compute_centroid, get_injection_hemisphere_id

from .model_data import ModelData #.model_data import ModelData
from .masks import Mask
from .utils import compute_centroid, get_injection_hemisphere_id
#from .utils import get_matrices


class VoxelDataset:
    """

    Parameters
    ----------
    sid: structure id

    Notes
    _____
    This class holds a collection of experiments in a particular structure
    """
    def __init__(self):#, sid):
        pass
        #self.sid = sid

def get_centroid(experiment):
    """Returns experiment centroid"""
    return experiment.centroid

def get_injection(experiment, normalized_injection):
    # print('ts',experiment.normalized_injection)
    """Returns experiment injection masked & flattened"""
    injection = experiment.get_injection(normalized_injection)
    return experiment.injection_mask.mask_volume(injection)

def get_projection(experiment, normalized_projection):
    """Returns experiment projection masked & flattened"""
    projection = experiment.get_projection(normalized_projection)
    return experiment.projection_mask.mask_volume(projection)

def yield_experiments(experiments):
    ev = experiments.values()
    keys = np.asarray(list(experiments.keys()))
    for i in range(len(keys)):
        yield (experiments[keys[i]])

def get_matrices(experiments):
    get_data = lambda x: (get_centroid(x),
                          get_injection(x,True),
                          get_projection(x, True))
    arrays = map(get_data, yield_experiments(experiments))
    centroids, injections, projections = map(np.array, zip(*arrays))
    return(centroids, injections, projections)


def get_experiment(cache, eid,sid,default_structure_ids):
    experiment = Experiment.from_cache(cache, eid)
    hemisphere_id = experiment.injection_hemisphere_id
    if hemisphere_id == 1:
        experiment = experiment.flip()
#     if (experiment.injection_hemisphere_id == 3 or hemisphere_id == experiment.injection_hemisphere_id):
#        experiment = experiment
#     elif experiment.flip_experiments:
#        experiment = experiment.flip()

    experiment.injection_structure_ids = [sid]
    experiment.major_structure = sid
    experiment.projection_structure_ids = default_structure_ids
    experiment.projection_hemisphere_id = 3
    experiment.normalized_injection = True
    experiment.normalized_projection = True
    return (experiment)

def get_voxeldata_msvd(cache, sid,experiments_exclude,default_structure_ids,cre):
    '''

    :param cache:
    :param sid:
    :param experiments_exclude:
    :param default_structure_ids:
    :param cre:
    :return:
    '''
    voxel_data = ModelData(cache, sid)
    experiment_ids = voxel_data.get_experiment_ids(experiments_exclude=experiments_exclude, cre=cre)
    experiment_ids = np.asarray(list(experiment_ids))
    experiments = {}
    # print('h')

    # experiment
    injection_mask = Mask.from_cache(
        cache,
        structure_ids=[sid],
        hemisphere_id=2)
    projection_mask = Mask.from_cache(
        cache,
        structure_ids=default_structure_ids,
        hemisphere_id=3)

    #first we get the experiment
    #experiment.from_cache populates injection and projection density
    for eid in experiment_ids:
        # print(eid)
        experiments[eid] = get_experiment(cache, eid, sid,default_structure_ids)
        experiments[eid].projection_mask = projection_mask
        experiments[eid].injection_mask = injection_mask

    #then we get the matrices... uses experiment.get_injection and .get_projection
    VDs = VoxelDataset()
    VDs.sid = sid
    VDs.projection_mask = projection_mask
    VDs.injection_mask = injection_mask
    VDs.experiments = experiments
    VDs.centroids, VDs.injections, VDs.projections = get_matrices(experiments)
    return (VDs)

def _pull_grid_data(cache, experiment_id):
    """Pulls data volumes using VoxelModelCache object.

    Parameters
    ----------
    cache : VoxelModelCache or MouseConnectivityCache instance.
        Object used to pull grid data.

    experiment_id : int
        Experiment id of the experiment from which to pull grid data.

    Returns
    -------
    dict
        Container of relevant data volumes.
        See allensdk.core.mouse_connectivity_cache for description of volumes.

    Notes
    -----
    voxel_model_cache.get_<data_volume> returns a tuple (data_volume, meta_data).
    We only care about the data volume.

    """
    return {
        "data_mask" : cache.get_data_mask(experiment_id)[0],
        "injection_density" : cache.get_injection_density(experiment_id)[0],
        "injection_fraction" : cache.get_injection_fraction(experiment_id)[0],
        "projection_density" : cache.get_projection_density(experiment_id)[0]
    }


def _mask_data_volume(data_volume, data_mask, tolerance=0.0):
    """Masks a given data volume in place.

    Parameters
    ----------
    data_volume : array, shape (x_ccf, y_ccf, z_ccf)
        Data volume to be masked.

    data_mask : array, shape (x_ccf, y_ccf, z_ccf)
        data_mask for given experiment (values in [0,1])
        See allensdk.core.mouse_connectivity_cache for more info.

    tolerance : float, optional (default=0.0)
        tolerance with which to define bad voxels in data_mask.


    Returns
    -------
    data_volume
        data_volume parameter masked in place.

    """
    if data_volume.shape != data_mask.shape:
        raise ValueError("data_volume (%s) and data_mask (%s) must be the same "
                         "shape!" % (data_volume.shape, data_mask.shape))

    # mask data volume
    data_volume[data_mask < tolerance] = 0.0

    return data_volume


def _compute_true_injection_density(injection_density, injection_fraction, inplace=False):
    """Computes 'true' injection_density.

    Takes into consideration injection fracion (proportion of pixels in the
    annotated injection site).

    Parameters
    ----------
    injection_density : array, shape (x_ccf, y_ccf, z_ccf)
        injection_density data volume.

    injection_fraction : array, shape (x_ccf, y_ccf, z_ccf)
        injection_fraction data volume.

    inplace : boolean
        If True, overwrites injection_density parameter, else returns new array.

    Returns
    -------
    array, shape (x_ccf, y_ccf, z_ccf)
        'true' injection density : injection_density * injection_fraction
    """
    if injection_density.shape != injection_fraction.shape:
        raise ValueError("injection_density (%s) and injection_fraction "
                         "(%s) must be the same shape!"
                         % (injection_density.shape, injection_fraction.shape))

    if inplace:
        np.multiply(injection_density, injection_fraction, injection_density)
        return injection_density

    return np.multiply(injection_density, injection_fraction)


class Experiment(object):
    """Class containing the data from an anterograde injection

    Experiment conveniently compiles the relevant information from a given
    anterograde viral tracing experiment data.

    Parameters
    ----------
    voxel_model_cache : VoxelModelCache object
        This supplies the interface for pulling experimental data.

    experiment_id : int
        AllenSDK id assigned to given experiment

    Examples
    --------
    >>> from mcmodels.core import Experiment, VoxelModelCache
    >>> cache = VoxelModelCache()
    >>> eid = 100141273
    >>> exp = Experiment(voxel_model_cache, eid)
    >>> exp.injection_density.shape
    (132,80,114)
    """

    DEFAULT_DATA_MASK_TOLERANCE = 0.5

    @classmethod
    def from_cache(cls, cache, experiment_id, data_mask_tolerance=None):
        """Alternative constructor allowing for pulling grid data.

        Parameters
        ----------
        cache : VoxelModelCache or MouseConnectivityCache instance.
            Object used to pull grid data.

        experiment_id : int
            Experiment id of the experiment from which to pull grid data.

        data_mask_tolerance : float, optional (default = None)
            Tolerance with which to mask 'bad' data. data_mask array has values
            on the interval [0,1], where a nonzero element indicates a 'bad'
            voxel. If None is passed, the parameter defaults to
            DEFAULT_DATA_MASK_TOLERANCE (0.5).
        """
        if data_mask_tolerance is None:
            data_mask_tolerance = cls.DEFAULT_DATA_MASK_TOLERANCE

        try:
            # pull data
            data_volumes = _pull_grid_data(cache, experiment_id)
        except AttributeError:
            raise ValueError('cache must be a MouseConnectivityCache or '
                             'VoxelModelCache object, not %s' % type(cache))

        # compute 'true' injection density (inplace)
        _compute_true_injection_density(data_volumes["injection_density"],
                                        data_volumes["injection_fraction"],
                                        inplace=True)

        # mask data in place
        mask_func = partial(_mask_data_volume,
                            data_mask=data_volumes["data_mask"],
                            tolerance=data_mask_tolerance)

        injection_density = mask_func(data_volumes["injection_density"])
        projection_density = mask_func(data_volumes["projection_density"])

        return cls(injection_density, projection_density)


    def __init__(self, injection_density=None, projection_density=None):
        # assume numpy array
        if injection_density.shape != projection_density.shape:
            raise ValueError("injection_density (%s) and projection_density "
                             "(%s) must be the same shape!"
                             % (injection_density.shape, projection_density.shape))

        self.injection_density = injection_density
        self.projection_density = projection_density

    def __repr__(self):
        return '{0}(volume_shape={1})'.format(
            self.__class__.__name__, self.injection_density.shape)

    @property
    def injection_hemisphere_id(self):
        """Returns injection hemisphere"""
        return get_injection_hemisphere_id(self.injection_density, majority=True)

    @property
    def bilateral_injection(self):
        """Returns injection hemisphere"""
        return get_injection_hemisphere_id(self.injection_density) == 3

    @property
    def injection_volume(self):
        """Returns total injection volume = sum(injection_density)"""
        return self.injection_density.sum()

    @property
    def projection_volume(self):
        """Returns total projection volume = sum(projection_density)"""
        return self.projection_density.sum()

    @property
    def centroid(self):
        """Returns centroid of the injection density."""
        return compute_centroid(self.injection_density)

    @property
    def normalized_injection_density(self):
        """Returns injection density normalized to have unit sum."""
        return self.injection_density / self.injection_volume

    @property
    def normalized_projection_density(self):
        """Returns projection_density normalized by the total injection_density"""
        return self.projection_density / self.injection_volume

    def get_injection(self, normalized=False):
        if normalized:
            return self.normalized_injection_density
        return self.injection_density

    def get_projection(self, normalized=False):
        if normalized:
            return self.normalized_projection_density
        return self.projection_density

    def flip(self):
        """Reflects experiment along midline.

        Returns
        -------
        self - flipped experiment
        """
        self.injection_density = self.injection_density[..., ::-1]
        self.projection_density = self.projection_density[..., ::-1]

        return self
