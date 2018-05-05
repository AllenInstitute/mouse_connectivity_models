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


Assumptions
-----------
- Spatial smoothness within divisions: the connectivity is assumed to vary smoothly
  as a function of distance within each of the major brain divisions.
- No influence between divisions: the connectivity is allowed to be discontinuous
  at division boundaries. These major brain divisions are in fact physically
  separated by :term:`white matter`, supporting this assumption.


:class:`VoxelConnectivityArray` Class
---------------------------------------

Since the full voxel x voxel connectivity matrix is ~200,000 x ~400,000 elements,
it will mostlikely not fit in your memory. Luckily, the connectivity matrix has
low rank structure, and we can take advantage of this by only computing the
connectivty matrix on the fly, in the area we want to perform computation.

Loading the array
~~~~~~~~~~~~~~~~~
The easiest way to load the :class:`VoxelConnectivityArray` is through the
:class:`VoxelModelCache` object:

        >>> from mcmodels.core import VoxelModelCache
        >>> cache = VoxelModelCache(manifest_file='connectivity/voxel_model_manifest.json')
        >>> voxel_array, source_mask, target_mask = cache.get_voxel_connectivity_array()
        >>> voxel_array
        VoxelConnectivityArray(dtype=float32, shape=(226346, 448962))

This downloads and caches the underlying data locally. Additionally,
this loads the factorization of the connectivity matrix into memory (~1G total).

Retrieving values from the array
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

No part of the connectivity matrix is computed until the user asks for a value
or set of values:

        >>> # some given source/target voxels
        >>> source, target = 20353, 68902
        >>>
        >>> # we index the VoxelConnectivyArray object just like it is a numpy ndarray
        >>> connection_strength = voxel_array[source, target]
        >>>
        >>> # a row would be the bi-lateral connection strength from a given voxel
        >>> row = voxel_array[source]
        >>>
        >>> # a column would be the connection strength to a given voxel
        >>> # from each voxel in the right hemisphere
        >>> column = voxel_array[:, target]
        >>>
        >>> # indexing the VoxelConnectivityArray object returns numpy ndarray
        >>> type(row)
        np.ndarray

If one wishes to operate on the full matrix (not recommended unless you have >1TB RAM!),
computing the full matrix is similar to loading an `hdf5` file:

        >>> import sys
        >>> full_array = vox_array[:]
        >>> sys.getsizeof(full_array)
        BIG!!!!!!


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

        >>> voxel_array.T
        VoxelConnectivityArray(dtype=float32, shape=(448962, 226346))

**iterator methods**

In addition to being able to index :class:`VoxelConnectivityArray` as a ``numpy.ndarray``,
:class:`VoxelConnectivityArray` implements several iterating methods:

.. currentmodule:: mcmodels.models.voxel.VoxelConnectivityArray

- :meth:`iterrows` : yields single rows
- :meth:`itercolumns` : yields single columns
- :meth:`iterrows_blocked` : yields blocks of rows given the number of blocks.
- :meth:`itercolumns_blocked` : yields blocks of columns rows given the number of blocks.


.. currentmodule:: mcmodels.models.voxel

:class:`RegionalizedModel` class
--------------------------------

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

We can again use the :class:`VoxelModelCache` object to download and cache
regionalized voxel models:

        >>> # this returns a pandas dataframe
        >>> normalized_connection_density = cache.get_normalized_connection_density()

Alternatively, we could construct the :class:`RegionalizedModel` object using
our :class:`VoxelConnectivityArray` and our source/target :class:`Mask` objects:

        >>> from mcmodels.models.voxel import RegionalizedModel
        >>> # get set of summary structures
        >>> structure_tree = cache.get_structure_tree()
        >>> summary_structures = structure_tree.get_structures_by_set_id([165787189])[0]
        >>> # the new ccf does not have sturcture 934 as a structure id
        >>> structure_ids = [s['id'] for s in summary_structures if s['id'] != 934]
        >>> # get keys
        >>> source_key = source_mask.get_key(structure_ids=summary_structures)
        >>> target_key = target_mask.get_key(structure_ids=summary_structures)
        >>> regionalized_model = RegionalizedModel.from_voxel_array(
        ... voxel_array, source_key, target_key)
        >>> normalized_connection_density = regionalized_model.normalized_connection_density

.. topic:: References

        .. [Knox2018] Knox et al. 'High resolution data-driven model of the mouse
           connectome.' bioRxiv 293019; doi: https://doi.org/10.1101/293019
