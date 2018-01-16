# Authors: Joseph Knox josephk@alleninstitute.org
# License:

# NOTE :: cythonize _BaseMask.map_to_ccf

from __future__ import division
import numpy as np

def union_mask(mcc, structure_ids):
    """Returns the union of a set of structure masks.

    Parameters
    ----------
    mcc : allensdk.core.mouse_connectivity_cache.MouseConnectivityCache object
        This supplies the interface for pulling experimental data
        from the AllenSDK.

    structure_ids : array-like, optional, shape (n_structure_ids,)
        AllenSDK CCF Annotation stucture ids to be included in the model

    Returns
    -------
    C : array-like, shape=(x_ccf, y_ccf, z_ccf)
        Union of structure_masks from each structure_id
    """
    masks = [ mcc.get_structure_mask(structure_id)[0]
              for structure_id in structure_ids ]
    return np.logical_or.reduce(masks)

def union_key(mcc, structure_ids):
    # NOTE: TERRIBLE NAME
    """Union of keys.
    """

    # TODO: could check that structures are not decendents of eachother

    # fencepost
    sid = structure_ids[0]
    key = sid*mcc.get_structure_mask(sid)[0]

    if len(structure_ids) > 1:
        for sid in structure_ids:
            # structures should be non overlapping!!!!
            np.add(key, sid*mcc.get_structure_mask(sid)[0], key)

    return key

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
        self.hemishpere = hemisphere

    def _get_mask(self):
        """   """
        mask = union_mask(self.mcc, self.structure_ids)
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
            self._mask = self._get_mask()
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
    def _get_key(self):
        """   """
        key = union_key(self.mcc, self.structure_ids)
        midline = key.shape[2]//2

        if self.hemisphere == 1:
            # contra
            key[:,:,:midline] = False
        elif self.hemisphere == 2:
            # ipsi
            key[:,:,midline:] = False

        return key.flatten().nonzero()[0]

    @property
    def key(self):
        try:
            return self._key
        except AttributeError:
            self._key = self._get_key()
            return self._key


    @property
    def key(self):
        """Returns the key ... """


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

    # enforces R hemisphere injection
    hemisphere = 2

    def __init__(self, mcc, structure_ids):
        super(SourceMask, self).__init__(mcc, structure_ids, self.hemisphere)

class TargetMask(_BaseMask):
    """Mask for target
    """

    _hemi_map = {
        "ipsi" : 2, #right
        "contra" : 1, #left,
        "both" : 3
    }

    def __init__(self, mcc, structure_ids, hemisphere):
        try:
            self.hemisphere = self._hemi_map[hemisphere]
        except KeyError:
            if hemisphere in range(1,4):
                self.hemisphere = hemisphere
            else:
                raise ValueError("must pass {} or {} to hemisphere".format(
                    self._hemi_map.keys(), self._hemi_map.values()
                ))

        super(TargetMask, self).__init__(mcc, structure_ids, hemisphere)
