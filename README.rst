.. -*- mode: rst -*-


cell_class_specific_connectivity_models
=========================
This branch contains analyses of the cell-class specific connectivities from FILL.
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