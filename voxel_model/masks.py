# Authors: Joseph Knox josephk@alleninstitute.org
# License:

# NOTE :: cythonize _BaseMask.map_to_ccf

from __future__ import division
import numpy as np

# only R hemisphere for source
_HEMISPHERES = [1,2,3]
_SOURCE_HEMISPHERE = 2
_TARGET_HEMI_MAP = {
    "ipsi" : 2, #right
    "contra" : 1, #left
    "both" : 3
}

def union_mask(mcc, structure_ids, return_key=False):
    """Returns the union of a set of structure masks.

    Parameters
    ----------
    mcc : allensdk.core.mouse_connectivity_cache.MouseConnectivityCache object
        This supplies the interface for pulling experimental data
        from the AllenSDK.

    structure_ids : array-like, optional, shape (n_structure_ids,)
        AllenSDK CCF Annotation stucture ids to be included in the model

    return_key : bool, optional (default=False)
        Return mask as a key (each element in array is structure id) (True)
        or as binary mask (False)

    Returns
    -------
    union : array-like, shape=(x_ccf, y_ccf, z_ccf)
        Union of structure_masks from each structure_id
    """
    # TODO: could check that structures are not decendents of eachother

    def pull_mask(sid):
        """Pulls mask. helper"""
        mask = mcc.get_structure_mask(sid)[0]
        if return_key:
            np.multiply(mask, sid, mask)

        return mask

    # fencepost
    union = pull_mask(structure_ids[0])

    if len(structure_ids) > 1:
        for sid in structure_ids[1:]:
            mask = pull_mask(sid)

            # structures should be non overlapping!!!!
            np.add(union, mask, union)

    return union

class _BaseMask(object):
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

    def __init__(self, mcc, structure_ids, hemisphere):
        self.mcc = mcc
        self.structure_ids = structure_ids
        self.hemisphere = hemisphere

    def _get_mask(self, return_key):
        """   """
        mask = union_mask(self.mcc, self.structure_ids, return_key=return_key)

        midline = mask.shape[2]//2
        if self.hemisphere == 1:
            # contra
            mask[:,:,:midline] = False
        elif self.hemisphere == 2:
            # ipsi
            mask[:,:,midline:] = False

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

#     @property
#     def key(self):
#         """Returns nonzero indices of mask"""
#         return self.mask.flatten().nonzero()[0]

    @property
    def key(self):
        try:
            key = self._key
        except AttributeError:
            self._key = self._get_mask(return_key=True)
            key = self._key

        return key.flatten().nonzero()[0]

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
        for idx, val in zip(self.key, y):
            y_ccf[idx] = val

        return y_ccf.reshape(self.ccf_shape)

class SourceMask(_BaseMask):
    """Mask for source
    """
    def __init__(self, mcc, structure_ids):
        super(SourceMask, self).__init__(mcc, structure_ids, _SOURCE_HEMISPHERE)

class TargetMask(_BaseMask):
    """Mask for target
    """
    # TODO: change to classmethod constructor
    def __init__(self, mcc, structure_ids, hemisphere=3):
        if hemisphere not in _HEMISPHERES:
            try:
                hemisphere = _TARGET_HEMI_MAP[hemisphere]
            except KeyError:
                raise ValueError("must pass {} or {} to hemisphere".format(
                        _TARGET_HEMI_MAP.keys(), _HEMISPHERES
                    ))

        super(TargetMask, self).__init__(mcc, structure_ids, hemisphere)
