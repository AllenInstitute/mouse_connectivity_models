Examples
======================================

`build_model.py` : builds the voxel x voxel connectivity model  
`build_region_matrices.py` : integrates the voxel x voxel matrix over the summary structures\*  

\* `build_model.py` must be run before any other examples or analysis

VoxelArray
======================================

Since the full voxel x voxel connectivity matrix is ~200,000 x ~400,000 elements, it will mostlikely not fit in your memory. Luckily, the connectivity matrix has low rank structure, and we can take advantage of this by only computing the connectivty matrix on the fly, in the area we want to perform computation.

### Loading the array ###
`VoxelArray` has a variety of construction options, for instance:

```python
import os
from mcmodels.models.voxel import VoxelConnectivityArray

# assuming weights, nodes live in data/
weights_file = os.path.join(os.getcwd(), "data", "weights.npy")
nodes_file = os.path.join(os.getcwd(), "data", "nodes.npy")

# construct a VoxelArray object from .npy files
model = VoxelConnectivityArray.from_npy(weights_file, nodes_file)
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

If one wishes to operate on the full matrix (not recommended unless you have >1TB RAM!), computing the full matrix is similar to loading an `hdf5` file:

```python
full_array = model[:]
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

Using the Mask object
=============================================
Using the `Mask` object in conjunction with `VoxelArray` will allow you to perform meaningful analysis with the connectivity model.

### Initializing a Mask object ###
When working with the full connectivty matrix, first either load, or initialize the defualt `Mask` objects.

```python
from mcmodels.core.masks import Mask
from mcmodels.utils import get_mcc

# returns a MouseConnectivityCache instance from allensdk
mcc = get_mcc()

# defualt masks are constructed this way
source_mask = Mask(mcc, hemisphere=2)
target_mask = Mask(mcc, hemisphere=3)
```

### Use cases ###

#### find out to which structure a given row/column belongs ####
```python
# find source voxel with greatest total connectivity
source = np.argmax(model.sum(axis=1))

# get key (flat array with values cooresponding to structure_ids)
key = source_mask.get_key()
structure_id = key[source]

# get structure name using structure_tree object from allensdk
structure_tree = mcc.get_structure_tree()
structure_name = structure_tree.get_structures_by_id([structure_id])[0]["name"]

print("Most connected source voxel is located in {}".format(structure_name))

# also one can determine where in space the voxel is located
source_coordinates = source_mask.coordinates
voxel = source_coordinates[source]

print("Most connected source voxel is located at {}".format(voxel))
```

#### find out to which rows/columns a given structure belongs ####
```python
# say we want to look at VISl
structure_id = structure_tree.get_structures_by_acronym(["VISl"])[0]["id"]

# rows of interest 
row_idx = source_mask.get_structure_indices(structure_ids=[structure_id])

# connectivity matrix with sources in VISl
visl_connectivity = model[row_idx]
```

#### Map the connectivity of a voxel to ccf space ####
```python
# row of interest
row = model[5562]

# map to ccf space
connection_volume = source_mask.map_masked_to_annotation(row)
```
