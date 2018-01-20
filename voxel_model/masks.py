# Authors: Joseph Knox josephk@alleninstitute.org
# License:

# TODO :: incorp save/load into Mask class
# TODO :: cythonize _BaseMask.map_to_ccf

from __future__ import division
import pickle
import numpy as np
import operator as op

from functools import reduce

def save_mask(mask, filename):
    with open(filename, "wb") as fn:
        pickle.dump(mask, fn, pickle.HIGHEST_PROTOCOL)

def load_mask(filename):
    with open(filename, "rb") as fn:
        return pickle.load(fn)

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

    # only R hemisphere for source
    _HEMISPHERES = [1,2,3]

    def __init__(self, mcc, structure_ids, hemisphere=3):
        self.mcc = mcc
        self.reference_space = self.mcc.get_reference_space()
        self.structure_ids = structure_ids

        if hemisphere in self._HEMISPHERES:
            self.hemisphere = hemisphere
        else:
            raise ValueError("must one of", _HEMISPHERES)

    def _mask_to_hemisphere(self, mask):
        """masks to hemi"""
        # mask to hemisphere
        midline = mask.shape[2]//2
        if self.hemisphere == 1:
            # contra
            mask[:,:,midline:] = 0
        elif self.hemisphere == 2:
            # ipsi
            mask[:,:,:midline] = 0

        return mask

    def _get_mask(self, structure_ids=None, return_key=False):
        """ ...  """
        mask = self.reference_space.make_structure_mask(self.structure_ids,
                                                        direct_only=False)
        return self._mask_to_hemisphere(mask)

    @property
    def mask(self):
        try:
            return self._mask
        except AttributeError:
            self._mask = self._get_mask(return_key=False)
            return self._mask

    @property
    def annotation_shape(self):
        return self.reference_space.annotation.shape

    @property
    def coordinates(self):
        """Returns coordinates inside mask"""
        return np.argwhere(self.mask)

    @property
    def nonzero(self):
        """returns masked indices"""
        return self.mask.nonzero()

    def get_key(self, structure_ids=None):
        """Returns flattened annotation key.

        ...
        ...

        Parameters
        ----------
        """
        if structure_ids is None:
            structure_ids = self.structure_ids
            mask = self.mask
        else:
            # from allensdk.core.reference_space.make_structure_mask
            mask = self.reference_space.make_structure_mask(structure_ids,
                                                            direct_only=False)

        # get descendants
        descendant_ids = self.reference_space.structure_tree.descendant_ids(
            structure_ids
        )

        # do not want to overwrite annotation
        annotation = np.copy(self.reference_space.annotation)

        for structure_id, descendants in zip(structure_ids, descendant_ids):
            # set annotation equal to structure where it has descendants
            idx = np.isin(annotation, descendants)
            annotation[ idx ] = structure_id

        # mask annotation to only structure ids
        np.multiply(annotation, mask, annotation)

        # mask to hemisphere
        annotation = self._mask_to_hemisphere(annotation)

        # returned flattened
        return annotation[ annotation.nonzero() ]

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
