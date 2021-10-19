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
conda create -n 'mcm_class' python=3.7 -y
source activate mcm_class 
pip install scikit-learn==0.22.1
pip install pandas
pip install seaborn
pip install dill
pip install allensdk==1.3.0
pip install pygam
pip install openpyxl

source deactivate mcm_class
conda env remove --name mcm_class

Execution
=========================

The code consists of several steps
0)  Metadata acquisition
1)  Model validation
2)  Connectivity creation
3)  Figure generation

0.0) Generation of leaf identities

The important pieces of metadata are

1.0) Model validation

analyses/paper_final/modelvalidations/ELModel_LeafTarget_ProjNorm_final.ipynb
analyses/paper_final/modelvalidations/ELModel_LeafTarget_ProjNorm_cutoffremoved_final.ipynb
analyses/paper_final/modelvalidations/ELModel_LeafTarget_InjNorm_cutoffremoved_final.ipynb

2.0) Generation of leaf-leaf connectivities
This was done a cluster using
analyses/paper_final/connectivitycreation/ELModel_LeafTarget_ProjNorm_v7.py
2.1) Generation of summary-summary connectivities from leaf-leaf connectivities
analyses/paper_final/connectivitycreation/Leaf_to_Sum_0621.ipynb