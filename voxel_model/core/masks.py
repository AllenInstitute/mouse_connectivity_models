""""
Module containing Mask object and supporting functions
"""

# Authors: Joseph Knox josephk@alleninstitute.org
# License:

# TODO : finish Mask docstring (examples)
from __future__ import division
from functools import reduce

import operator as op
import numpy as np


def _validate_descendant_ids(structure_ids, descendant_ids):
    """Validates that descendant_ids are of the correct form."""
    if len(structure_ids) != len(descendant_ids):
        # descendant_ids are not generated from structure_ids
        return False

    return all([dids[0] == sid for sid, dids
                in zip(structure_ids, descendant_ids)])


def _check_disjoint_structures(structure_ids, descendant_ids):
    """Validates that structures are disjoint."""
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
    """Object for masking the grid data from allensdk.

    This object is useful for masking grid data as well as reshaping/filling
    'masked' arrays to be the shape of the annotaion (CCF) from allensdk. Also,
    this object is useful for determining the location or structure id of a
    given voxel from the voxel-voxel connectivity matrix.

    Parameters
    ----------
    mcc : allensdk.core.mouse_connectivity_cache.MouseConnectivityCache object
        This supplies the interface for pulling experimental data
        from the AllenSDK.

    structure_ids : array-like, optional, shape (n_structure_ids,)
        AllenSDK CCF Annotation stucture ids to be included in the model

    Hemisphere : int
        hemisphere id to be included in the projection in the model.
            * 1, left hemisphere
            * 2, right hemisphere
            * 3, both hemispheres

    Attributes
    ----------
    referece_space : reference_space object
        see allensdk.reference_space

    Examples
    --------
    >>> from voxel_model.masks import Mask
    >>>

    """

    DEFAULT_STRUCTURE_IDS = tuple([8])

    def __init__(self, mcc, structure_ids=None, hemisphere=3):

        if structure_ids is None:
            structure_ids = self.DEFAULT_STRUCTURE_IDS

        # update reference space to include only assigned voxels
        reference_space = mcc.get_reference_space()
        reference_space.remove_unassigned(update_self=True)

        self.mcc = mcc
        self.structure_ids = structure_ids
        self.hemisphere = hemisphere
        self.reference_space = reference_space
        self.annotation = reference_space.annotation
        self.structure_tree = reference_space.structure_tree

    @staticmethod
    def _mask_to_hemisphere(mask, hemisphere):
        """Masks a given data volume to a hemisphere."""
        # mask to hemisphere
        midline = mask.shape[2]//2
        if hemisphere == 1:
            mask[:, :, midline:] = 0

        elif hemisphere == 2:
            mask[:, :, :midline] = 0

        return mask

    def _get_mask(self, structure_ids, hemisphere=None):
        """Gets mask property (boolean array)"""
        if hemisphere is None:
            hemisphere = self.hemisphere

        mask = self.reference_space.make_structure_mask(structure_ids,
                                                        direct_only=False)
        return Mask._mask_to_hemisphere(mask, hemisphere)

    @property
    def mask(self):
        """Boolean mask defining 'interesting' voxels."""
        try:
            return self._mask
        except AttributeError:
            self._mask = self._get_mask(self.structure_ids)
            return self._mask

    def _get_assigned_structures(self):
        # return flattened set of list of lists
        descendants = self.structure_tree.descendant_ids(self.structure_ids)
        return set(reduce(op.add, descendants, []))

    @property
    def assigned_structures(self):
        """List of resolved structures in annotation"""
        try:
            return self._assigned_structures
        except AttributeError:
            self._assigned_structures = self._get_assigned_structures()
            return self._assigned_structures

    @property
    def annotation_shape(self):
        """Shape of the annotation array (CCF)"""
        return self.annotation.shape

    @property
    def coordinates(self):
        """Returns coordinates inside mask."""
        return np.argwhere(self.mask)

    @property
    def masked_shape(self):
        """Shape a data volume would become after masking."""
        return (np.count_nonzero(self.mask),)

    def get_structure_flattened_mask(self, structure_ids=None, hemisphere=None):
        """
        ...
        """
        if structure_ids is None:
            structure_ids = self.structure_ids

        if structure_ids is self.structure_ids and hemisphere is None:
            # saves time if already computed
            mask = self.mask
        else:
            mask = self._get_mask(structure_ids, hemisphere=hemisphere)

        # mask this mask to self.mask
        return self.mask_volume(mask)

    def get_flattened_voxel_index(self, voxel_idx):
        """Return the index of a givel voxel in the flattened mask.

        Parameters
        ----------
        voxel_idx : array
            coordinates of voxel of which the flattened index position is wanted

        Returns
        -------
        int
            idx of voxel_idx in flattened mask
        """

        idx = np.where((self.coordinates == voxel_idx).all(axis=1))[0]

        try:
            # return int from array
            return idx[0]
        except IndexError as e:
            # array is empty
            raise ValueError("voxel index is not in mask.coordinates"
                             "\nIndexError\n %s" % e)


    def get_structure_indices(self, structure_ids=None, hemisphere=None):
        """
        ...
        """
        aligned = self.get_structure_flattened_mask(structure_ids, hemisphere)

        return aligned.nonzero()[0]

    def get_key(self, structure_ids=None, hemisphere=None):
        # TODO: look into cleaning up check for disjoint
        """Returns flattened annotation key.

        Useful in performing structure specific computations on the voxel-voxel
        array.

        Parameters
        ----------
        structure_ids : list, optional (defalut=None)
            Ids of structures which to include in the key. If None, the
            structure_ids used to make the Mask object will be used.

        hemisphere : int, optional (defalut=None)
            Hemisphere to include in the key. If None, the hemisphere used
            to maske the Mask object will be used.

        Returns
        -------
        key = array, shape (masked_shape,), type np.int
            Key mapping an element in a masked data volume to its structure id
            in the annotation. Each element in key is a structure_id.

        """
        # do not want to overwrite annotation
        annotation = self.annotation.copy()

        if structure_ids is None and hemisphere is None:
            # return key of all resolved structures in annotation
            annotation[np.logical_not(self.mask)] = 0
            return self.mask_volume(annotation)

        # get list of descendant_ids for each structure id
        descendant_ids = self.structure_tree.descendant_ids(structure_ids)

        if not _check_disjoint_structures(structure_ids, descendant_ids):
            raise ValueError("structures are not disjoint")

        for structure_id, descendants in zip(structure_ids, descendant_ids):

            if len(descendants) > 1:
                # set annotation equal to structure where it has descendants
                idx = np.isin(annotation, descendants)
                annotation[idx] = structure_id

        # get mask according to args
        mask = self._get_mask(structure_ids, hemisphere=hemisphere)

        # mask to structure_ids
        annotation[np.logical_not(mask)] = 0

        return self.mask_volume(annotation)

    def mask_volume(self, X):
        """Masks a given volume.

        Paramters
        ---------
        X - array, shape (x_ccf, y_ccf, z_ccf)
            Data volume to be masked. Must be same shape as
            self.annotation_shape

        Returns
        -------
        array - shape (masked_shape)
            Masked data volume.

        """
        if X.shape != self.annotation_shape:
            raise ValueError("X must be same shape as annotation")

        return X[self.mask.nonzero()]

    def fill_volume_where_masked(self, X, fill, inplace=True):
        """Fills a data volume where mask is valid.

        Paramters
        ---------
        X : array, shape (x_ccf, y_ccf, z_ccf)
            Array to be filled where mask is valid.

        fill : int or array, shape=(masked_shape)
            Fill value or array.

        inplace : boolean
            If True, X is filled in place, else a copy is returned.

        Returns
        -------
        array - shape (x_ccf, y_ccf, z_ccf)
            Filled array.
        """

        if X.shape != self.annotation_shape:
            raise ValueError("X must be same shape as annotation")

        _X = X.copy() if not inplace else X

        # numpy throws value error if type(fill)=array && fill.shape != idx
        _X[self.mask.nonzero()] = fill
        return _X

    def map_masked_to_annotation(self, y):
        """Maps a masked vector y back to annotation volume.

        Paramters
        ---------
        y : array, shape (masked_shape)
            Array to be mapped into where mask is valid.

        Returns
        -------
        y_volume : shape (x_ccf, y_ccf, z_ccf)
            Array same shape as annotation, filled with input parameter y where
            mask is valid

        """
        # indices where y
        idx = self.mask.nonzero()

        if y.shape != idx[0].shape:
            # TODO : better error statement
            raise ValueError("Must be same shape as key")

        y_volume = np.zeros(self.annotation_shape)
        y_volume[idx] = y

        return y_volume
