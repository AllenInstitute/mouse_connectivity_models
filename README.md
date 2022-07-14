# Cell-class specific connectivity models

This branch contains analyses of cell-class specific connectivities.
These analyses consist of construction of the connectivities via the expected-loss model, factorization of the estimated connectivity matrices, establishing a limit of detection, and more.
The workflow for these steps to reproduce the results in the paper is as follows.

# Installation

Follow the instructions in the master branch at <http://AllenInstitute.github.io/mouse_connectivity_models/>.
Then run

```
conda create -n 'mouse_connectivity' python=3.7 -y
source activate mouse_connectivity
pip install scikit-learn==0.22.1 pandas seaborn dill allensdk==1.3.0 pygam openpyxl
```

Add the custom `utils/validation` and `decomposition/_nmf` files to your sklearn path.

# Summary

The code to reproduce results from the paper consists of

1.  Model validation
2.  Connectivity creation
3.  Analysis and figure generation

We use the following pieces of metadata

```
data/meta/input_011520.json
data/meta/new_manifest.json
data/meta/experiments_exclude.json
analyses/paper_final/metadata/major_structures.npy
analyses/paper_final/metadata/major_structure_ids.npy
data/meta/Whole Brain Cre Image Series_curation only.xlsx
analyses/paper_final/metadata/ontological_order_leaves_v3.npy
```

These may also be obtained from the Allen SDK and `analyses/paper_final/get_info/get_info.ipynb`.

# Model validation

We evaluate the performance of candidate connectivity estimators in

```
analyses/paper_final/modelvalidations/ELModel_LeafTarget_ProjNorm_final.ipynb
analyses/paper_final/modelvalidations/ELModel_LeafTarget_InjNorm_cutoffremoved_final.ipynb
analyses/paper_final/modelvalidations/ELModel_LeafTarget_ProjNorm_cutoffremoved_final.ipynb
```

The first notebook contains cross-validation (CV) results using projection signals normalized by the norms of the projections themselves.
The second contains CV results using projection signals normalized by the norm of the injections, with a small fraction of experiments removed due to outlying small injection norm.
The third contains CV results using the projection-normalized signals with the same set of experiments removed.

The cross-validation results from the first notebook are used to estimate the expected-loss surface and Nadaraya-Watson bandwidths used in estimation of the connectivity matrices in the next stage.
These notebooks were run locally.

# Connectivity creation

Leaf to leaf connectivity estimation was performed using the estimators generated in the previous step in

```
analyses/paper_final/connectivitycreation/ELModel_LeafTarget_ProjNorm_v7.sh
analyses/paper_final/connectivitycreation/ELModel_LeafTarget_ProjNorm_v7.py
```

Once generated, these can be combined into summary structure to summary structure connectivities as in

```
analyses/paper_final/connectivitycreation/Leaf_to_Sum_0621.ipynb
```

# Figures and analysis

Model and connectivity analyses and plotting were performed in the following notebooks:

```
analyses/paper_final/figures/Figure1_SuppFigure1.ipynb
analyses/paper_final/figures/Figure2.ipynb
analyses/paper_final/figures/Figure3.ipynb
analyses/paper_final/figures/Figure4_SuppFig.ipynb
```
