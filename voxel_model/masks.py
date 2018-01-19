# Authors: Joseph Knox josephk@alleninstitute.org
# License:

# TODO :: cythonize _BaseMask.map_to_ccf

from __future__ import division
import numpy as np

# only R hemisphere for source
_HEMISPHERES = [1,2,3]

class Mask(object):
    """Base Mask class for SourceMask and TargetMask.

    ...

    Parameters
    ----------
    mcc : allensdk.core.mouse_connectivity_cache.MouseConnectivityCache object
        This supplies the interface for pulling experimental data
        from the AllenSDK.

    structure_ids : array-like, optional, shape (n_structure_ids,)
        AllenSDK CCF Annotation stucture ids to be included in the model

    Hemisphere : int or str, optional (default 3)
        hemisphere id to be included in the projection in the model.
            * 1, "contra" : left hemisphere - contralateral
            * 2, "ipsi" : right hemisphere - ipsilateral
            * 3, "both" : both hemispheres - full-brain projection

    Attributes
    ----------
    """

    def __init__(self, mcc, structure_ids, hemisphere=3, other_mask=None):
        self.mcc = mcc
        self.structure_ids = structure_ids

        if hemisphere in _HEMISPHERES:
            self.hemisphere = hemisphere
        else:
            raise ValueError("must one of", _HEMISPHERES)

        if other_mask is None:
            self.other_mask = other_mask
        else:
            if isinstance(other_mask, Mask):
                # we only care about mask property
                self.other_mask = other_mask.mask
            else:
                raise ValueError("if other_mask, must be mask object!")


    def _get_mask(self, return_key=False):
        """   """
        # annotation is a key?
        if return_key:
            # each elem in mask : structure id
            mask = self.mcc.get_annotation_volume()[0]
        else:
            # binary mask
            reference_space = self.mcc.get_reference_space()
            mask = reference_space.make_structure_mask( self.structure_ids,
                                                        direct_only=False )

        # mask to hemisphere
        midline = mask.shape[2]//2
        if self.hemisphere == 1:
            # contra
            mask[:,:,:midline] = 0
        elif self.hemisphere == 2:
            # ipsi
            mask[:,:,midline:] = 0

        # mask to additional mask
        if self.other_mask is not None:
            # allow for mask intersection
            # intersection = np.logical_and(mask, self.other_mask)
            mask[ np.logical_not(self.other_mask).nonzero() ] = 0

        return mask

    @property
    def mask(self):
        try:
            return self._mask
        except AttributeError:
            self._mask = self._get_mask(return_key=False)
            return self._mask

    @property
    def ccf_shape(self):
        return self.mask.shape

    @property
    def coordinates(self):
        """Returns coordinates inside mask"""
        return np.argwhere(self.mask)

    @property
    def where(self):
        """returns masked indices"""
        return self.mask.nonzero()

    @property
    def key(self):
        """Returns flattened otology key"""
        try:
            return self._key
        except AttributeError:
            key = self._get_mask(return_key=True)
            self._key = key[ key.nonzero() ]
            return self._key

    def map_to_ccf(self, y):
        """Maps a masked vector y back to ccf

        Paramters
        ---------
        y
        Returns
        -------
        y_ccf
        """
        if y.shape != self.key.shape:
            raise ValueError("Must be same shape as key")

        y_ccf = np.zeros(np.prod(self.ccf_shape))

        # SLOW!!! (cythonize???)
        for idx, val in zip(self.coordinates, y):
            y_ccf[idx] = val

        return y_ccf.reshape(self.ccf_shape)
