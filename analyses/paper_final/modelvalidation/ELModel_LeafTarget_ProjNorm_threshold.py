%load_ext autoreload
%autoreload 2

import os
import numpy as np
import pandas as pd 
import sys
from sklearn.metrics import pairwise_distances
import math
import dill as pickle
import matplotlib.pyplot as plt
import seaborn as sns

workingdirectory = os.popen('git rev-parse --show-toplevel').read()[:-1]
sys.path.append(workingdirectory)
os.chdir(workingdirectory)

import allensdk.core.json_utilities as ju
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache

from mcmodels.core import VoxelModelCache
from mcmodels.core.connectivity_data import get_connectivity_data
from mcmodels.core.utils import get_ordered_summary_structures, get_leaves_ontologicalorder, get_indices_2ormore, get_eval_indices
from mcmodels.models.expectedloss.crossvalidation import get_loss_surface_cv_spline
from mcmodels.models.expectedloss.crossvalidation import get_embedding_cv
from mcmodels.models.voxel.crossvalidation import CrossvalNW

#read data
TOP_DIR = workingdirectory
INPUT_JSON = workingdirectory + '/data/meta/input_011520.json'
EXPERIMENTS_EXCLUDE_JSON = workingdirectory + '/data/meta/experiments_exclude.json'
input_data = ju.read(INPUT_JSON)
experiments_exclude = ju.read(EXPERIMENTS_EXCLUDE_JSON)
manifest_file = input_data.get('manifest_file')
manifest_file = os.path.join(TOP_DIR, manifest_file)
cache = VoxelModelCache(manifest_file=manifest_file)
st = cache.get_structure_tree()
ai_map = st.get_id_acronym_map()
ia_map = {value: key for key, value in ai_map.items()}
major_structures = np.load(workingdirectory + '/data/meta/major_structures.npy')
major_structure_ids = np.load(workingdirectory + '/data/meta/major_structure_ids.npy')
data_info = pd.read_excel(workingdirectory + '/data/meta/Whole Brain Cre Image Series_curation only.xlsx', 'all datasets curated_070919pull')
data_info.set_index("id", inplace=True)
with open('data/meta/leafs.pickle', 'rb') as handle:
    leafs = pickle.load(handle)
ontological_order_leaves = np.load(workingdirectory + '/data/meta/ontological_order_leaves_v3.npy')
COARSE_STRUCTURE_SET_ID = 2
DEFAULT_STRUCTURE_SET_IDS = tuple([COARSE_STRUCTURE_SET_ID])
tree = cache.get_structure_tree()
default_structures = tree.get_structures_by_set_id(DEFAULT_STRUCTURE_SET_IDS)
default_structure_ids = [st['id'] for st in default_structures if st['id'] != 934]

import pickle
with open('analyses/results/EL_model_060622_leafleaf.pickle', 'rb') as handle:
    twostage_leaf_creleaf2 = pickle.load(handle)
    

gammas = np.asarray([0.0001,0.001,0.01,0.1,1,10,100,1000,10000])

gammaids = np.zeros(12)
for m in range(12):
    gammaids[m] = np.where(gammas == twostage_leaf_creleaf2.bestgamma_weighted[m])[0]
gammaids = np.asarray(gammaids , dtype = int)

self = twostage_leaf_creleaf2
threshes = np.asarray([0,1e-9,1e-8,1e-7,1e-6,1e-5, 1e-4, 1e-3,1e-2])
sel_gammas = twostage_leaf_creleaf2.bestgamma_weighted

models = self.models
nmodels = len(models)
predictions = self.predictions
data = self.data
nt = len(threshes)
eval_indices = self.eval_indices
results_n1 = np.zeros((nmodels,nt))
results_p1 = np.zeros((nmodels,nt))
bestfpfn = np.zeros(nmodels)
npts = np.zeros(nmodels)
#fpfn_proportion = np.zeros((nmodels,nt))
for m in range(nmodels):
    sid = models[m]
    inds = eval_indices[sid]
    npt = len(eval_indices[sid])
    npts[m] = npt
#     aboves[sid] = np.zeros((nt, npt))
#     belows[sid] = np.zeros((nt, npt))
#     w_ab[sid] = np.zeros((nt, npt))
#     w_be[sid] = np.zeros((nt, npt))
    
    #results_p1 = np.zeros(nt)
    for c in range(nt):
        output = np.ones(data[sid][inds].shape)
        baseline = np.ones(data[sid][inds].shape)
        #pred_mat = loocvpredictions[315][5].copy()
        zeroind_data = np.asarray(np.where(data[sid][inds] == 0.))
        print(sid, m , inds, c, predictions[sid].shape)
        zeroind_pred = np.asarray(np.where(predictions[sid][gammaids[m]][inds]  <= threshes[c]))
        output[tuple(zeroind_pred)] = 0.
        baseline[tuple(zeroind_data)] = 0.
        diff = output - baseline
        # results_p1 is how many false positives
        # results_n1 is how many false negatives
        results_n1[m,c] = np.where(diff == -1)[0].shape[0]
        results_p1[m,c] = np.where(diff == 1)[0].shape[0]
    
    bestfpfn[m] = np.abs(results_p1[m] - results_n1[m]).argmin()
    
fpfn_proportion = np.abs(results_p1 - results_n1) / (np.expand_dims(npts,1)*data[512].shape[1])
fpfn_proportion= pd.DataFrame(fpfn_proportion, columns=threshes, index=major_structures)
combpos = fpfn_proportion + fpfn_proportion
    #return(fpfn_proportion)
    #return(results_n1)
#     512 0 [ 7 11 21 31] 0 (9, 36, 1123)

fig = plt.figure(figsize=(20,15))
ax1 = plt.subplot2grid((20,20), (0,0), colspan=19, rowspan=19)
ax2 = plt.subplot2grid((20,20), (19,0), colspan=19, rowspan=1)
ax3 = plt.subplot2grid((20,20), (0,19), colspan=1, rowspan=19)
#ax4 = plt.subplot2grid((20,20), (19,19), colspan=19, rowspan=1)
#ax4.set_title('asdf')
#ax3 = plt.subplot2grid((20,20), (0,19), colspan=1, rowspan=19)
#mask = np.zeros_like(combpos)
#mask[np.tril_indices_from(mask)] = True

sns.heatmap(combpos, ax = ax1, annot=True, cmap="Greys", linecolor='b', cbar = False)
#ax1.set_title("Threshold", fontsize = 15)
#ax1.ax_cbar.set_position([1.0,0.4,.1,.1])
ax1.set_xticklabels(ax1.get_xticklabels(),fontsize = 20)
ax1.set_yticklabels(ax1.get_yticklabels(),fontsize = 20, rotation = 0)
#ax1.set_xlabel('Threshold')
ax1.xaxis.tick_top()
ax1.set_xticklabels(combpos.columns,rotation=40)

sns.heatmap((pd.DataFrame(combpos.mean(axis=0))).transpose(), ax=ax2,  annot=True, cmap="Greys", cbar=False, xticklabels=False, yticklabels=False)
sns.heatmap(pd.DataFrame(combpos.mean(axis=1)), ax=ax3,  annot=True, cmap="Greys", cbar = False,
            xticklabels=False, yticklabels=False)


plt.suptitle('False negatives + false positives', fontsize = 40)
fig.savefig('paper/KoelleConn_fromlocal_0623/figs/Threshold', pad_inches = 0)