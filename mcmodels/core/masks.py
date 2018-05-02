""""
Module containing Mask object and supporting functions.
"""
# Authors: Joseph Knox <josephk@alleninstitute.org>
# License: Allen Institute Software License

from __future__ import division
from functools import reduce
import operator as op

import numpy as np
from allensdk.core import json_utilities


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
    'masked' arrays to be the shape of the annotation (CCF) from allensdk. Also,
    this object is useful for determining the location or structure id of a
    given voxel from the voxel-voxel connectivity matrix.

    Parameters
    ----------
    voxel_model_cache : VoxelModelCache object
        This supplies the interface for pulling cached reference space,
        annotation, and structure tree objects.

    structure_ids : array-like, optional, shape (n_structure_ids,)
        AllenSDK CCF Annotation structure ids to be included in the model

    Hemisphere : int
        hemisphere id to be included in the projection in the model.
            * 1, left hemisphere
            * 2, right hemisphere
            * 3, both hemispheres

    Attributes
    ----------
    reference_space : reference_space object
        see allensdk.reference_space

    Examples
    --------
    >>> from mcmodels.core import Mask, VoxelModelCache
    >>> cache = VoxelModelCache()
    >>> mask = Mask.from_cache(cache)
    >>> # what shape will a masked volume be
    >>> mask.masked_shape
    (448962,)
    """

    GREY_STRUCTURE_ID = 8
    DEFAULT_STRUCTURE_IDS = tuple([GREY_STRUCTURE_ID])

    BILATERAL_HEMISPHERE_ID = 3
    DEFAULT_HEMISPHERE_ID = BILATERAL_HEMISPHERE_ID

    @classmethod
    def from_cache(cls, cache, **kwargs):
        """

        Parameters
        ----------
        cache : caching object
            MCC or RSC or VMC
        """
        try:
            reference_space = cache.get_reference_space()
        except AttributeError:
            raise ValueError('Must pass a MouseConnectivtyCache, '
                             'ReferenceSpaceCache, or VoxelModelCache object '
                             'not %s' % type(cache))

        return cls(reference_space, **kwargs)

    def __init__(self, reference_space, structure_ids=None, hemisphere_id=None):
        if structure_ids is None:
            structure_ids = self.DEFAULT_STRUCTURE_IDS
        if hemisphere_id is None:
            hemisphere_id = self.DEFAULT_HEMISPHERE_ID

        # update reference space to include only assigned voxels
        reference_space.remove_unassigned(update_self=True)

        self.reference_space = reference_space
        self.structure_ids = structure_ids
        self.hemisphere_id = hemisphere_id

    def __repr__(self):
        if len(self.structure_ids) > 3:
            structure_ids = "{0}, ..., {1}".format(self.structure_ids[0],
                                                   self.structure_ids[-1])
        else:
            structure_ids = ", ".join(map(str, self.structure_ids))

        return "{0}(hemisphere_id={1}, structure_ids=[{2}])".format(
            self.__class__.__name__, self.hemisphere_id, structure_ids)

    @staticmethod
    def _mask_to_hemisphere(mask, hemisphere_id):
        """Masks a given data volume to a hemisphere."""
        # mask to hemisphere
        midline = mask.shape[2]//2
        if hemisphere_id == 1:
            mask[..., midline:] = 0

        elif hemisphere_id == 2:
            mask[..., :midline] = 0

        return mask

    def _get_mask(self, structure_ids, hemisphere_id=None):
        """Gets mask property (boolean array)"""
        if hemisphere_id is None:
            hemisphere_id = self.hemisphere_id

        mask = self.reference_space.make_structure_mask(structure_ids,
                                                        direct_only=False)
        return Mask._mask_to_hemisphere(mask, hemisphere_id)

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
        descendants = self.reference_space.structure_tree.descendant_ids(self.structure_ids)
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
    def coordinates(self):
        """Returns coordinates inside mask."""
        return np.argwhere(self.mask)

    @property
    def masked_shape(self):
        """Shape a data volume would become after masking."""
        return (np.count_nonzero(self.mask),)

    def get_structure_flattened_mask(self, structure_ids=None, hemisphere_id=None):
        """Masks a structure_mask or union of structure_masks.

        Parameters
        ----------
        structure_ids : array-like, optional (default = None)
            A list of structure ids with which to construct a structure union
            mask. If None, the object's structure_ids are used.

        hemisphere_id : int, optional (default = None)
            The hemisphere to which the structure union mask will be additionally
            masked to. If None, the object's hemisphere_id is used.

        Returns
        -------
        array - shape (masked_shape)
            Masked structure union mask.
        """
        if structure_ids is None:
            structure_ids = self.structure_ids

        if structure_ids is self.structure_ids and hemisphere_id is None:
            # saves time if already computed
            mask = self.mask
        else:
            mask = self._get_mask(structure_ids, hemisphere_id=hemisphere_id)

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
            raise ValueError("voxel index %s is not in mask.coordinates"
                             "\nIndexError\n %s" % (voxel_idx, e))


    def get_structure_indices(self, structure_ids=None, hemisphere_id=None):
        """Returns the indices of a masked structure_mask or union of structure_masks.

        Parameters
        ----------
        structure_ids : array-like, optional (default = None)
            A list of structure ids with which to construct a structure union
            mask. If None, the object's structure_ids are used.

        hemisphere_id : int, optional (default = None)
            The hemisphere to which the structure union mask will be additionally
            masked to. If None, the object's hemisphere_id is used.

        Returns
        -------
        array - shape (masked_nonzero,)
            The indices of the masked structure union mask that are nonzero
        """
        aligned = self.get_structure_flattened_mask(structure_ids, hemisphere_id)

        return aligned.nonzero()[0]

    def get_key(self, structure_ids=None, hemisphere_id=None):
        # TODO: look into cleaning up check for disjoint
        """Returns flattened annotation key.

        Useful in performing structure specific computations on the voxel-voxel
        array.

        Parameters
        ----------
        structure_ids : list, optional (default=None)
            Ids of structures which to include in the key. If None, the
            structure_ids used to make the Mask object will be used.

        hemisphere_id : int, optional (default=None)
            Hemisphere to include in the key. If None, the hemisphere used
            to mask the Mask object will be used.

        Returns
        -------
        key = array, shape (masked_shape,), type np.int
            Key mapping an element in a masked data volume to its structure id
            in the annotation. Each element in key is a structure_id.
        """
        # do not want to overwrite annotation
        annotation = self.reference_space.annotation.copy()

        if structure_ids is None and hemisphere_id is None:
            # return key of all resolved structures in annotation
            annotation[np.logical_not(self.mask)] = 0
            return self.mask_volume(annotation)

        # get list of descendant_ids for each structure id
        descendant_ids = self.reference_space.structure_tree.descendant_ids(structure_ids)

        if not _check_disjoint_structures(structure_ids, descendant_ids):
            raise ValueError("structures %s are not disjoint" % structure_ids)

        for structure_id, descendants in zip(structure_ids, descendant_ids):

            if len(descendants) > 1:
                # set annotation equal to structure where it has descendants
                idx = np.isin(annotation, descendants)
                annotation[idx] = structure_id

        # get mask according to args
        mask = self._get_mask(structure_ids, hemisphere_id=hemisphere_id)

        # mask to structure_ids
        annotation[np.logical_not(mask)] = 0

        return self.mask_volume(annotation)

    def mask_volume(self, X):
        """Masks a given volume.

        Parameters
        ----------
        X - array, shape (x_ccf, y_ccf, z_ccf)
            Data volume to be masked. Must be same shape as
            self.reference_space.annotation.shape

        Returns
        -------
        array - shape (masked_shape)
            Masked data volume.
        """
        if X.shape != self.reference_space.annotation.shape:
            raise ValueError("X must be same shape as annotation: %s (not %s)"
                             % (self.reference_space.annotation.shape, X.shape))

        return X[self.mask.nonzero()]

    def fill_volume_where_masked(self, X, fill, inplace=True):
        """Fills a data volume where mask is valid.

        Parameters
        ----------
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
        if X.shape != self.reference_space.annotation.shape:
            raise ValueError("X must be same shape as annotation: %s (not %s)"
                             % (self.reference_space.annotation.shape, X.shape))

        _X = X.copy() if not inplace else X

        # numpy throws value error if type(fill)=array && fill.shape != idx
        _X[self.mask.nonzero()] = fill
        return _X

    def map_masked_to_annotation(self, y):
        """Maps a masked vector y back to annotation volume.

        Parameters
        ----------
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
            raise ValueError("Must be same shape as key: %s (not %s)"
                             % (idx[0].shape, y.shape))

        y_volume = np.zeros(self.reference_space.annotation.shape)
        y_volume[idx] = y

        return y_volume
