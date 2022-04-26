cell_class_specific_connectivity_models
=========================
This branch contains analyses of the cell-class specific connectivities.
These analyses consist of construction of the connectivities via the expected-loss model, factorization of the estimated connectivity matrices, establishing a limit of detection, and more.
The workflow for these steps to reproduce the results in the paper is as follows.

Installation
=========================
Follow the instructions in the master branch at <http://AllenInstitute.github.io/mouse_connectivity_models/>.

Configuration
=========================
`
conda create -n 'mcm_class' python=3.7 -y
source activate mcm_class 
pip install scikit-learn==0.22.1
pip install pandas
pip install seaborn
pip install dill
pip install allensdk==1.3.0
pip install pygam
pip install openpyxl

Add the custom utils/validation and decomposition/_nmf files to sklearn
source deactivate mcm_class
conda env remove --name mcm_class
`
Execution
=========================

The code consists of several steps
0)  Metadata acquisition
1)  Model validation
2)  Connectivity creation
3)  Figure generation

0.0) Generation of leaf identities

The important pieces of metadata are
```
data/data_final/input_011520.json
```
input config.  Contains manifest_file
```
data/data_final/new_manifest.json
```
manifest file

```
data/data_final/experiments_exclude.json
```
experiments to exclude
```
manifest_file = input_data.get('manifest_file')
manifest_file = os.path.join(TOP_DIR, manifest_file)

analyses/paper_final/metadata/major_structures.npy
major_structures

analyses/paper_final/metadata/major_structure_ids.npy
major_structure_ids

data/data_final/Whole Brain Cre Image Series_curation only.xlsx
```
contains experimental metadata
```
analyses/paper_final/metadata/ontological_order_leaves_v3.npy
```
contains the leaves in ontological order for Allen CCF V3



1.0) Model validation
```
analyses/paper_final/modelvalidations/ELModel_LeafTarget_ProjNorm_final.ipynb
analyses/paper_final/modelvalidations/ELModel_LeafTarget_ProjNorm_cutoffremoved_final.ipynb
analyses/paper_final/modelvalidations/ELModel_LeafTarget_InjNorm_cutoffremoved_final.ipynb
```
2.0) Generation of leaf-leaf connectivities
This was done a cluster using
```
analyses/paper_final/connectivitycreation/ELModel_LeafTarget_ProjNorm_v7.py
```
2.1) Generation of summary-summary connectivities from leaf-leaf connectivities
```
analyses/paper_final/connectivitycreation/Leaf_to_Sum_0621.ipynb
```
