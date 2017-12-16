"""
NOTE :: cythonize _BaseMask.map_to_ccf
"""
from __future__ import division
import numpy as np

class _BaseMask(object):
    """Base Mask class
    """

    def __init__(self, mcc, structure_ids, hemisphere):
        self.mcc = mcc
        self.structure_ids = structure_ids
        self.hemishpere = hemisphere

    @staticmethod
    def union_mask(structure_ids):
        """Returns the union of a set of structure masks"""
        masks = [ mcc.get_structure_mask(structure_id)[0]
                  for structure_id in structure_ids ]
        return np.reduce.logical_or(masks)

    @property(self):
    def _get_mask(self):
        """   """
        _mask = union_mask(self.structure_ids)

        if self.hemisphere == 1:
            # contra
            self._mask[:,:,:self.ccf_shape[2]/2] = np.unint(0)
        elif self.hemisphere == 2:
            # ipsi
            self._mask[:,:,self.ccf_shape[2]/2:] = np.unint(0)

        return self._mask

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
    def key(self):
        """Returns nonzero indices of mask"""
        return self.mask.flatten().nonzero()[0]

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

        y_ccf = np.zeros(np.multiply(*self.ccf_shape))

        # SLOW!!! (cythonize???)
        for idx, val in zip(self.key, y):
            y_ccf[idx] = val

        return y_ccf.reshape(y_ccf)


class SourceMask(_BaseMask):
    """Mask for source
    """

    # enforces R hemisphere injection
    hemisphere = 2

    def __init__(self, mcc, structure_ids):
        super(SourceMask, self).__init__(mcc, structure_ids)

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
            self.hemisphere = _hemi_map[hemisphere]
        except KeyError:
            if hemisphere in range(3):
                self.hemisphere = hemisphere
            else:
                raise ValueError("must pass {} or {} to hemisphere".format(
                    _hemi_map.keys(), _hemi_map.values()
                ))

        super(TargetMask, self).__init__(mcc, structure_ids, hemisphere)
