# Authors: Joseph Knox josephk@alleninstitute.org
# License:

from __future__ import division
from functools import reduce
import pickle
import numpy as np
import operator as op

__all__ = [
    "Mask"
]

def _validate_descendant_ids(structure_ids, descendant_ids):
    """ ... """
    # check lengths (for zip)
    if len(structure_ids) != len(descendant_ids):
        return False

    return all( [dids[0] == sid for sid, dids
                  in zip(structure_ids, descendant_ids)] )

def _check_disjoint_structures(structure_ids, descendant_ids):
    """ checks that descendants are disjoint """

    if _validate_descendant_ids(structure_ids, descendant_ids):
        # first elem in descendant_ids is structure id from which they descend
        only_descendants = [ids[1:] for ids in descendant_ids if len(ids) > 1]

        if set(structure_ids) & set(reduce(op.add, only_descendants, [])):
            # a structure_id is a descendant of another
            return False

        # else: descendant_ids == structure_ids, assume @ bottom of annotation
        return True

    return False

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
    VALID_HEMISPHERES = [1,2,3]

    def _check_hemisphere(self, hemisphere):
        """ ..."""
        if hemisphere not in self.VALID_HEMISPHERES:
            raise ValueError("must one of", self.VALID_HEMISPHERES)

        return hemisphere

    def __init__(self, mcc, structure_ids, hemisphere=3):
        self.mcc = mcc
        self.structure_ids = structure_ids
        self.hemisphere = self._check_hemisphere(hemisphere)

        # get reference_space module and update to resolved structures
        self.reference_space = self.mcc.get_reference_space()
        self.reference_space.remove_unassigned(update_self=True)

        # get updated ref space data structures
        self.structure_tree = self.reference_space.structure_tree
        self.annotation = self.reference_space.annotation

    @staticmethod
    def _mask_to_hemisphere(mask, hemisphere):
        """masks to hemi"""
        # mask to hemisphere
        midline = mask.shape[2]//2
        if hemisphere == 1:
            # contra
            mask[:,:,midline:] = 0
        elif hemisphere == 2:
            # ipsi
            mask[:,:,:midline] = 0

        return mask

    def _get_mask(self, structure_ids, hemisphere=None):
        """ ...  """
        if hemisphere is not None:
            hemisphere = self._check_hemisphere( hemisphere )
        else:
            hemisphere = self.hemisphere

        mask = self.reference_space.make_structure_mask( structure_ids,
                                                         direct_only=False )
        return Mask._mask_to_hemisphere( mask, hemisphere )

    @property
    def mask(self):
        try:
            return self._mask
        except AttributeError:
            self._mask = self._get_mask(self.structure_ids)
            return self._mask

    @property
    def annotation_shape(self):
        return self.annotation.shape

    @property
    def coordinates(self):
        """Returns coordinates inside mask"""
        return np.argwhere(self.mask)

    @property
    def masked_shape(self):
        return ( np.count_nonzero(self.mask), )

    def get_key(self, structure_ids=None, hemisphere=None):
        # TODO: look into cleaning up check for disjoint
        """Returns flattened annotation key.

        ...
        ...

        Parameters
        ----------
        """
        if structure_ids is None:
            structure_ids = self.structure_ids

        # get list of descendant_ids for each structure id
        descendant_ids = self.structure_tree.descendant_ids( structure_ids )

        if not _check_disjoint_structures( structure_ids, descendant_ids ):
            raise ValueError("structures are not disjoint")

        if structure_ids is self.structure_ids and hemisphere is None:
            # saves time if already computed
            mask = self.mask
        else:
            mask = self._get_mask( structure_ids, hemisphere=hemisphere )

        # do not want to overwrite annotation
        annotation = self.annotation.copy()

        for structure_id, descendants in zip(structure_ids, descendant_ids):

            if len(descendants) > 1:
                # set annotation equal to structure where it has descendants
                idx = np.isin(annotation, descendants)
                annotation[ idx ] = structure_id

        # mask to structure_ids
        annotation[ np.logical_not(mask) ] = 0

        return self.mask_volume(annotation)

    def mask_volume(self, X):
        """Masks a given volume

        Paramters
        ---------
        X
        Returns
        -------
        y
        """

        if X.shape != self.annotation_shape:
            # TODO : better error statement
            raise ValueError("X must be same shape as annotation")

        return X[ self.mask.nonzero() ]

    def fill_volume_where_masked(self, X, fill, inplace=True):
        """Masks a given volume

        Paramters
        ---------
        X
        Returns
        -------
        y
        """

        if X.shape != self.annotation_shape:
            # TODO : better error statement
            raise ValueError("X must be same shape as annotation")

        _X = X.copy() if not inplace else X

        # numpy throws value error if type(fill)=array && fill.shape != idx
        _X[ self.mask.nonzero() ] = fill
        return _X

    def map_masked_to_annotation(self, y):
        """Maps a masked vector y back to annotation volume

        Paramters
        ---------
        y
        Returns
        -------
        y_ccf
        """
        # indices where y
        idx = self.mask.nonzero()

        if y.shape != idx[0].shape:
            # TODO : better error statement
            raise ValueError("Must be same shape as key")

        y_volume = np.zeros(self.annotation_shape)
        y_volume[ idx ] = y

        return y_volume

    def save(self, filename):
        """ ... """
        with open(filename, "wb") as fn:
            pickle.dump(self, fn, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, filename):
        """ ... """
        with open(filename, "rb") as fn:
            return pickle.load(fn)
