# Authors: Joseph Knox josephk@alleninstitute.org
# License:

from __future__ import division
import pickle
import six
import operator as op
import numpy as np

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

    def __init__(self, mcc, structure_ids, hemisphere=3):
        self.mcc = mcc
        self.structure_ids = structure_ids

        if hemisphere in self.VALID_HEMISPHERES:
            self.hemisphere = hemisphere
        else:
            raise ValueError("must one of", self.VALID_HEMISPHERES)

        # get reference_space module and update to resolved structures
        self.reference_space = self.mcc.get_reference_space()
        self.reference_space.remove_unassigned(update_self=True)

        # get updated ref space data structures
        self.structure_tree = self.reference_space.structure_tree
        self.annotation = self.reference_space.annotation

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
        return self.annotation.shape

    @property
    def coordinates(self):
        """Returns coordinates inside mask"""
        return np.argwhere(self.mask)

    def get_key(self, structure_ids=None, disjoint_structures=True):
        # TODO: look into cleaning up check for disjoint
        """Returns flattened annotation key.

        ...
        ...

        Parameters
        ----------
        """
        # do not want to overwrite annotation
        annotation = np.copy(self.annotation)

        if structure_ids is None:
            # use structure_ids mask was originally built with
            structure_ids = self.structure_ids
            mask = self.mask
        else:
            # from allensdk.core.reference_space.make_structure_mask
            # want only non overlapping structures (annotation)
            mask = self.reference_space.make_structure_mask(structure_ids,
                                                            direct_only=False)
        # get list of descendant_ids for each structure id
        # NOTE : descendant_ids includes the structure id
        descendant_ids = self.structure_tree.descendant_ids( structure_ids )

        if disjoint_structures:
            all_descendants = [ids[1:] for ids in descendant_ids if len(ids) > 1]

            if all_descendants:
                all_descendants = set(reduce(op.add, all_descendants))

                if set(structure_ids) & all_descendants:
                    raise ValueError("structure_ids are not disjoint!")

            for structure_id, descendants in zip(structure_ids, descendant_ids):
                if len(descendants) > 1:
                    # set annotation equal to structure where it has descendants
                    idx = np.isin(annotation, descendants)
                    annotation[ idx ] = structure_id

            # mask to structure_ids
            annotation[ np.logical_not(mask) ] = 0

            return self.mask_volume(annotation)

        else:
            # would have to iterate through structure_ids hierarchically
            raise NotImplementedError

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
