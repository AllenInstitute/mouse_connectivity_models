Examples
======================================

`build_model.py` : builds the voxel x voxel connectivity model  
`build_region_matrices.py` : integrates the voxel x voxel matrix over the summary structures\*  
`voxel_array.py` : displays the functionality of the `VoxelArray` object (see below)

\* `build_model.py` must be run before any other examples or analysis

voxel_array.py
======================================

Since the full voxel x voxel connectivity matrix is ~200,000 x ~400,000 elements, it will mostlikely not fit in your memory. Luckily, the connectivity matrix has low rank structure, and we can take advantage of this by only computing the connectivty matrix on the fly, in the area we want to perform computation.

### Loading the array ###
`VoxelArray` has a variety of construction options, for instance:

```python
import os
from voxel_model.voxel_array import VoxelArray

# assuming weights, nodes live in data/
weights_file = os.path.join(os.getcwd(), "data", "weights.npy")
nodes_file = os.path.join(os.getcwd(), "data", "nodes.npy")

# construct a VoxelArray object from .npy files
model = VoxelArray.from_npy(weights_file, nodes_file)
```

This loads the factorization of the connectivity matrix into memory (~1G total).

### Retrieving values from the array ###
No part of the connectivity matrix is computed until the user asks for a value or set of values:

```python
# some given source/target voxels
source, target = 20353, 68902

# we index the VoxelArray object just like it is a numpy ndarray
connection_strength = model[source, target]

# a row would be the bi-lateral connection strength from a given voxel
row = model[source]

# a column would be the connection strength to a given voxel
# from each voxel in the right hemisphere
column = model[:, target]

# indexing the VoxelArray object returns numpy ndarray
print( type(row) )
```

### VoxelArray methods ###
`VoxelArray` also has a few methods implemented from `numpy.ndarray`. These include:
* `dtype`
* `shape`
* `size`
* `T`, `transpose`
* `astype`
* `sum`
* `mean`

and are called just like their `numpy.ndarray` counter parts:
```python
transposed = model.T
```
