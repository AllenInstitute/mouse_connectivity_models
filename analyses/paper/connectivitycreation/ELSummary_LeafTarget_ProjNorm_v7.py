import os
import numpy as np
import pandas as pd 
import sys
#import pickle
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
from sklearn.metrics import pairwise_distances
from sklearn.kernel_ridge import KernelRidge
import math
import dill as pickle

workingdirectory = os.popen('git rev-parse --show-toplevel').read()[:-1]
sys.path.append(workingdirectory)
os.chdir(workingdirectory)


import allensdk
import allensdk.core.json_utilities as ju
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache

from mcmodels.utils import get_aligned_ids
from mcmodels.core import VoxelModelCache
from mcmodels.connectivity.creation import get_connectivity_matrices3
from mcmodels.core.connectivity_data import get_connectivity_data
#from mcmodels.models.voxel.crossvalidation import get_nwloocv_predictions_multimodel_merge_dists
from mcmodels.utils import nonzero_unique
from mcmodels.core.utils import get_structure_id,get_ordered_summary_structures, get_leaves_ontologicalorder, get_indices_2ormore, get_indices, get_indices2,get_eval_indices,screen_index_matrices,screen_index_matrices2,screen_index_matrices3
from mcmodels.regressors import NadarayaWatson
from mcmodels.core.plotting import plot_loss_surface,plot_loss_scatter
from mcmodels.models.expectedloss.crossvalidation import get_loss_surface_cv_spline
from mcmodels.models.expectedloss.crossvalidation import get_embedding_cv
from mcmodels.models.crossvalidation import get_best_hyperparameters,get_loss_best_hyp,get_loss, Crossval
from mcmodels.models.voxel.crossvalidation import get_nwloocv_predictions_multimodel_merge_dists

def get_row_col_names(connectivity_data, target_ordering):
    
    rnames = np.asarray([ia_map[target_ordering[i]] for i in range(len(target_ordering))])
    ipsi_target_regions = connectivity_data.ipsi_target_regions
    contra_target_regions = connectivity_data.contra_target_regions                               
    ipsi_indices= np.asarray([])
    contra_indices = np.asarray([])
    for iy in target_ordering: 
        ipsi_indices = np.concatenate([ipsi_indices, np.where(ipsi_target_regions==iy)[0]] )
        contra_indices = np.concatenate([contra_indices, np.where(contra_target_regions==iy)[0]] )
    ipsi_indices = np.asarray(ipsi_indices, dtype = int)   
    contra_indices = np.asarray(contra_indices, dtype = int)    
    reorder = np.concatenate([ipsi_indices, len(ipsi_indices) + contra_indices])  
    colids = np.concatenate([ipsi_target_regions, contra_target_regions])[reorder]
    cnames = np.asarray([ia_map[colids[i]] for i in range(len(colids))])
    ccomb = np.vstack([np.concatenate([np.repeat('ipsi',connectivity_data.ipsi_target_regions.shape[0]),
                                       np.repeat('contra',connectivity_data.contra_target_regions.shape)]), cnames])
    ccomb = np.asarray(ccomb)
    tuples2 = list(zip(*ccomb))
    cnam_multi = pd.MultiIndex.from_tuples(tuples2, names=['first', 'second'])
    return(cnam_multi, rnames)

#read data
data_dir = workingdirectory + '/data/rawdata/'
INPUT_JSON = data_dir + 'input_011520.json'
EXPERIMENTS_EXCLUDE_JSON = data_dir + 'experiments_exclude.json'
input_data = ju.read(INPUT_JSON)
experiments_exclude = ju.read(EXPERIMENTS_EXCLUDE_JSON)
#manifest_file = input_data.get('manifest_file')
#manifest_file = os.path.join(data_dir, manifest_file)
manifest_file = data_dir + '/new_manifest.json'
cache = VoxelModelCache(manifest_file=manifest_file)
st = cache.get_structure_tree()
ai_map = st.get_id_acronym_map()
ia_map = {value: key for key, value in ai_map.items()}
major_structures = np.load(workingdirectory + '/paper/info/major_structures.npy')
major_structure_ids = np.load(workingdirectory + '/paper/info/major_structure_ids.npy')
data_info = pd.read_excel(data_dir +'/Whole Brain Cre Image Series_curation only.xlsx', 'all datasets curated_070919pull')
data_info.set_index("id", inplace=True)

with open(workingdirectory + '/data/info/leafs.pickle', 'rb') as handle:
    leafs = pickle.load(handle)

ontological_order_leaves = np.load(workingdirectory + '/paper/info/ontological_order_leaves_v3.npy')
ontological_order = np.load(workingdirectory + '/paper/info/ontological_order_v3.npy')
ontological_order_leaves_majors = get_aligned_ids(st,ontological_order_leaves,major_structure_ids)
ontological_order_leaves_summary = get_aligned_ids(st,ontological_order_leaves,ontological_order)

connectivity_data = get_connectivity_data(cache, major_structure_ids, experiments_exclude, remove_injection = False, structure_set_id=167587189)
connectivity_data.get_injection_hemisphere_ids()
connectivity_data.align()
connectivity_data.get_centroids()
connectivity_data.get_data_matrices(major_structure_ids)

connectivity_data.ai_map = ai_map
connectivity_data.get_crelines(data_info)
connectivity_data.get_summarystructures(data_info)
connectivity_data.summary_structures = {sid: connectivity_data.structure_datas[sid].summary_structures for sid in major_structure_ids}#get_indices_2ormore(connectivity_data.leafs)
connectivity_data.leafs = leafs

sid0 = list(connectivity_data.structure_datas.keys())[0]
#Identify keys denoting which voxels correspond to which structure in the ipsi and contra targets.
targ_ord = ontological_order_leaves
source_ord = ontological_order_leaves
contra_targetkey = connectivity_data.structure_datas[sid0].projection_mask.get_key(structure_ids=targ_ord, hemisphere_id=1)
#ontological_order = np.load('/Users/samsonkoelle/alleninstitute/sambranch/mouse_connectivity_models/paper/info/ontological_order_v3.npy')
ipsi_targetkey = connectivity_data.structure_datas[sid0].projection_mask.get_key(structure_ids=targ_ord, hemisphere_id=2)
connectivity_data.get_regionalized_normalized_data(source_ord, ipsi_targetkey, contra_targetkey)
summary_structures = {sid: connectivity_data.structure_datas[sid].summary_structures for sid in major_structure_ids}#get_indices_2ormore(connectivity_data.leafs)

#el_leafsurf_leafsmth_v3_leafleaf_

experiment_sids_surfaces = leafs
experiment_sids_nws = leafs
model_ordering = ontological_order_leaves_majors
source_ordering_surface = ontological_order_leaves_summary
source_ordering_nw = ontological_order_leaves
source_ordering = ontological_order_leaves
target_ordering = ontological_order_leaves
for sid in major_structure_ids:
    connectivity_data.structure_datas[sid].crelines = connectivity_data.creline[sid]
    
with open(workingdirectory + '/paper/trainedmodels/ELleaf_surface_0427_leafleaf2.pickle', 'rb') as handle:
    surfaces = pickle.load(handle)
 
#source_reg = np.asarray(['MOp2/3', 'MOp5', 'MOp6a','MOs2/3', 'MOs5', 'MOs6a' ])
eval_cre_list_old = ['C57BL/6J', 'Cux2-IRES-Cre','Ntsr1-Cre_GN220','Rbp4-Cre_KL100','Tlx3-Cre_PL56']
eval_cre_list = np.unique(np.concatenate(list(connectivity_data.creline.values())))
eval_cre_list = np.setdiff1d(eval_cre_list,eval_cre_list_old)
cnam_multi, rnames = get_row_col_names(connectivity_data, ontological_order_leaves)
eval_cre_names =  ['C57BL6J', 'Cux2-IRES-Cre','Ntsr1-Cre_GN220','Rbp4-Cre_KL100','Tlx3-Cre_PL56']
#eval_cre_list

for c in range(len(eval_cre_list)):
    print(c, eval_cre_list[c])
    conn_v3 = get_connectivity_matrices3(connectivity_data, surfaces, experiment_sids_surfaces,experiment_sids_nws, model_ordering, source_ordering_surface, source_ordering_nw, source_ordering, target_ordering, [eval_cre_list[c]])
    connectivity_matrices = pd.DataFrame(conn_v3[0], columns = cnam_multi, index=rnames)
    connectivity_matrices.to_csv(workingdirectory + '/paper/connectivities/el_sumsurf_leafsmth_leafleaf_' + str(eval_cre_names[c]) + '0428.csv')
    
