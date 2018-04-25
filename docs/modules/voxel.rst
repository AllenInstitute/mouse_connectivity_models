.. _voxel:

===============================================
New Voxel-scale Connectivity Model [Knox2018]_
===============================================

.. currentmodule:: mcmodels.models.voxel

The voxel-scale model from [Knox2018]_ is the first full brain voxel-scale of the
mouse connectome. The model performs :ref:`Nadaraya-Watson regression
<nadaraya_watson>` to infer the connectivity between each of the voxels in
the brain into 12 :term:`major brain divisions` to each of the voxels in the
whole brain. The source space is split between these major brain divisions as to
prevent influence from injections performed into adjacent brain divisions.

:class:`VoxelModel` implements MORE


Assumptions
-----------
- Spatial smoothness within divisoins: the connectivity is assumed to vary smoothly
  as a function of distance within each of the major brain divisions.
- No influence between divisions: the connectivity is allowed to be discontinuous
  at divison boundaries. These major brain divisions are in fact physically
  separated by :term:`white matter`, supporting this assumption.


:class:`VoxelConnectivityArray` Class
---------------------------------------

Since the full voxel x voxel connectivity matrix is ~200,000 x ~400,000 elements,
it will mostlikely not fit in your memory. Luckily, the connectivity matrix has
low rank structure, and we can take advantage of this by only computing the
connectivty matrix on the fly, in the area we want to perform computation.

Loading the array
~~~~~~~~~~~~~~~~~
:class:`VoxelConnectivityArray` has a variety of construction options, for instance

        >>> import os
        >>> from mcmodels.models.voxel import VoxelConnectivityArray
        >>>
        >>> # assuming weights, nodes live in data/
        >>> weights_file = os.path.join(os.getcwd(), "data", "weights.npy")
        >>> nodes_file = os.path.join(os.getcwd(), "data", "nodes.npy")
        >>>
        >>> # construct a VoxelArray object from .npy files
        >>> vox_array = VoxelConnectivityArray.from_npy(weights_file, nodes_file)
        >>> vox_array.shape
        (xxx, xxx)

This loads the factorization of the connectivity matrix into memory (~1G total).

Retrieving values from the array
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

No part of the connectivity matrix is computed until the user asks for a value
or set of values:

        >>> # some given source/target voxels
        >>> source, target = 20353, 68902
        >>>
        >>> # we index the VoxelConnectivyArray object just like it is a numpy ndarray
        >>> connection_strength = vox_array[source, target]
        >>>
        >>> # a row would be the bi-lateral connection strength from a given voxel
        >>> row = vox_array[source]
        >>>
        >>> # a column would be the connection strength to a given voxel
        >>> # from each voxel in the right hemisphere
        >>> column = vox_array[:, target]
        >>>
        >>> # indexing the VoxelConnectivityArray object returns numpy ndarray
        >>> type(row)
        np.ndarray

If one wishes to operate on the full matrix (not recommended unless you have >1TB RAM!),
computing the full matrix is similar to loading an `hdf5` file:

        >>> import sys
        >>> full_array = vox_array[:]
        >>> sys.getsizeof(full_array)
        xxxxxx



VoxelConnectivityArray methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**numpy.ndarray like methods**

:class:`VoxelConnectivityArray` also has a few methods implemented from ``numpy.ndarray``.
These include:

.. currentmodule:: mcmodels.models.voxel.VoxelConnectivityArray

- :attr:`dtype`
- :attr:`shape`
- :attr:`size`
- :attr:`T`
- :meth:`transpose`
- :meth:`astype`
- :meth:`sum`
- :meth:`mean`

and are called just like their ``numpy.ndarray`` counter parts:

        >>> transposed = model.T
        >>> transposed.shape
        (xxx, xxx)

**initialization methods**

.. currentmodule:: mcmodels.models.voxel

Additionally, :class:`VoxelConnectivityArray` implements several loading methods
to support loading the array from:

- :meth:`VoxelConnectivityArray.from_hdf5`
- :meth:`VoxelConnectivityArray.from_npy`
- :meth:`VoxelConnectivityArray.from_csv`


Also, an :class:`VoxelConnectivityArray` instance can be constructed from a fitted
:class:`.VoxelModel` object:

- :meth:`VoxelConnectivityArray.from_fitted_voxel_model`


**iterator methods**

In addition to being able to index :class:`VoxelConnectivityArray` as a ``numpy.ndarray``,
:class:`VoxelConnectivityArray` implements several iterating methods:

.. currentmodule:: mcmodels.models.voxel.VoxelConnectivityArray

- :meth:`iterrows` : yields single rows
- :meth:`itercolumns` : yields single columns
- :meth:`iterrows_blocked` : yields blocks of rows given the number of blocks.
- :meth:`itercolumns_blocked` : yields blocks of columns rows given the number of blocks.

:class:`RegionalizedModel` class
--------------------------------

.. currentmodule:: mcmodels.models.voxel.RegionalizedModel

Our voxel-scale model can be regionalized as well by integrating the connectivity
matrix over some parcellation.

Metrics
~~~~~~~

Given a parcellation, integrating the connectivity
over source and target regions gives us :term:`connection strength`. Since the
relative sizes of the regions may be vastly different, we can normalize this metric
by dividing the connection strength by the size of ether the target region
(:term:`connection density`), the size of the source region (:term:`normalized
connection strength`) or by both the source and target (:term:`normalized
connection density`).

Using the parcellation defined in the Allen 3D Reference Atlas and a set of
structures, we can easily regionalize our voxel-scale connectivity:

TODO: need to make this more fool-proof, VoxelConnectivityArray should have
source/target mask generation

        >>> from mcmodels.models.voxel import RegionalizedModel
        >>> from mcmodels.core.mask import Mask
        >>> from mcmodels.utils import get_mcc
        >>>
        >>> # returns a MouseConnectivityCache instance with some default settings
        >>> mcc = get_mcc()
        >>>
        >>> source_mask = Mask(mcc, hemisphere=2)
        >>> target_mask = Mask(mcc, hemisphere=3)
        >>>
        >>> # get set of summary structures
        >>> summary_structures = source_mask.structure_tree.get_structures_by_set_id([165787189])[0]
        >>>
        >>> # get keys
        >>> source_key = source_mask.get_key(structure_ids=summary_structures)
        >>> target_key = target_mask.get_key(structure_ids=summary_structures)
        >>>
        >>> region_weights = RegionalizedModel.from_voxel_array(vox_array, source_key, target_key)
        >>> region_weights.normalized_connection_density.shape
        (292, 292)



.. topic:: References

        .. [Knox2018] "A high resolution data-driven model of the mouse connectome"
