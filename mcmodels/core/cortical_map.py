# Authors: Joseph Knox <josephk@alleninstitute.org>
# License: Allen Institute Software License

import os

import h5py
import nrrd
import numpy as np


class CorticalMap(object):
    '''Maps image volume values to cortical surface.

    NOTE: Only works using 100 micron image volumes with shape (132, 80, 114)

    Parameters
    ----------
    projection : string, optional (default : 'top_view')
        Type of surface mapping to project onto. Options are 'top_veiw' and
        'dorsal_flatmap'.

    References
    ----------
    `CCF Technical Whitepaper <http://help.brain-map.org/display/
    mouseconnectivity/Documentation?preview=/2818171/10813534/
    Mouse_Common_Coordinate_Framework.pdf>`_
    '''

    REFERENCE_SHAPE = (132, 80, 114)
    VALID_PROJECTIONS = ('top_view', 'dorsal_flatmap')

    @staticmethod
    def _load_paths(projection):
        module_name = os.path.dirname(__file__)
        paths_dir = os.path.join(module_name, 'cortical_coordinates')
        path = os.path.join(paths_dir, '%s_paths_100.h5' % projection)
        with h5py.File(path, 'r') as f:
            view_lookup = f['view lookup'][:]
            paths = f['paths'][:]

        return view_lookup, paths

    def __init__(self, projection='top_view'):
        if projection not in self.VALID_PROJECTIONS:
            raise ValueError('projection must be one of %s,  not %s'
                             % (self.VALID_PROJECTIONS, projection))

        self.view_lookup, self.paths = self._load_paths(projection)
        self.projection = projection

    def transform(self, volume, agg_func=np.mean, fill_value=0):
        '''Transforms image volume values to 2D cortical surface.

        Paramters
        ---------
        volume : np.ndarray, shape (132, 80, 114)
            Image volume to transform. Must be same shape as reference space.

        agg_func : callable, optional (default : np.mean)
            Aggregation function with which to apply along projection paths.

        fill_value : float, optional (default : 0)
            Value to fill result outside of cortical surface.

        Returns
        -------
        result : np.ndarray, shape (132, 114) 'top_view' or (136, 272) 'dorsal_flatmap'
            Projection of the image volume onto the surface of the cortex.
        '''
        def apply_along_path(i):
            path = self.paths[i]
            arr = volume.flat[path[path.nonzero()]]
            return agg_func(arr)

        if volume.shape != self.REFERENCE_SHAPE:
            raise ValueError('volume must have shape %s, not %s'
                             % (self.REFERENCE_SHAPE, volume.shape))
        if not callable(agg_func):
            raise ValueError('agg_func must be callable, not %s' % agg_func)

        # initialize output
        result = np.zeros(self.view_lookup.shape, dtype=volume.dtype)
        apply_along_paths_ = np.vectorize(apply_along_path)

        idx = np.where(self.view_lookup > -1)
        result[idx] = apply_along_paths_(self.view_lookup[idx])

        if fill_value != 0:
            result[self.view_lookup == -1] = fill_value

        return result
