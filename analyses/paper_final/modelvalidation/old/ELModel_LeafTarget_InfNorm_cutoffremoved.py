print("starting model validation")

import os
import numpy as np
import pandas as pd
import sys
from sklearn.metrics import pairwise_distances
import math
import dill as pickle
import matplotlib.pyplot as plt
import seaborn as sns
import allensdk.core.json_utilities as ju

workingdirectory = os.popen("git rev-parse --show-toplevel").read()[:-1]
sys.path.append(workingdirectory)
os.chdir(workingdirectory)

from mcmodels.core import VoxelModelCache
from mcmodels.core.connectivity_data import get_connectivity_data
from mcmodels.core.utils import (
    get_indices_2ormore,
    get_eval_indices,
)
from mcmodels.models.expectedloss.crossvalidation import get_loss_surface_cv_spline
from mcmodels.models.expectedloss.crossvalidation import get_embedding_cv
from mcmodels.models.expectedloss.plotting import plot_loss_surface, plot_loss_scatter
from mcmodels.models.voxel.crossvalidation import CrossvalNW
from mcmodels.models.plotting import plot_loss

# load metadata
TOP_DIR = workingdirectory
INPUT_JSON = workingdirectory + "/data/meta/input_011520.json"
EXPERIMENTS_EXCLUDE_JSON = workingdirectory + "/data/meta/experiments_exclude.json"
COARSE_STRUCTURE_SET_ID = 2
DEFAULT_STRUCTURE_SET_IDS = tuple([COARSE_STRUCTURE_SET_ID])
FOLDER = workingdirectory + "/data/raw/"

input_data = ju.read(INPUT_JSON)
experiments_exclude = ju.read(EXPERIMENTS_EXCLUDE_JSON)
manifest_file = input_data.get("manifest_file")
manifest_file = os.path.join(TOP_DIR, manifest_file)
cache = VoxelModelCache(manifest_file=manifest_file)
tree = cache.get_structure_tree()
st = cache.get_structure_tree()
ai_map = st.get_id_acronym_map()
ia_map = {value: key for key, value in ai_map.items()}
major_structures = np.load(workingdirectory + "/data/meta/major_structures.npy")
major_structure_ids = np.load(workingdirectory + "/data/meta/major_structure_ids.npy")
data_info = pd.read_excel(
    workingdirectory + "/data/meta/Whole Brain Cre Image Series_curation only.xlsx",
    "all datasets curated_070919pull",
)
data_info.set_index("id", inplace=True)
with open("data/meta/leafs.pickle", "rb") as handle:
    leafs = pickle.load(handle)
ontological_order_leaves = np.load(
    workingdirectory + "/data/meta/ontological_order_leaves_v3.npy"
)
ontological_order = np.load("data/meta/ontological_order_v3.npy")
default_structures = tree.get_structures_by_set_id(DEFAULT_STRUCTURE_SET_IDS)
default_structure_ids = [st["id"] for st in default_structures if st["id"] != 934]
reorder = np.asarray([4, 7, 2, 1, 10, 9, 11, 3, 5, 8, 6, 0], dtype=int)

connectivity_data = get_connectivity_data(
    cache,
    major_structure_ids,
    experiments_exclude,
    remove_injection=False,
    structure_set_id=167587189,
    folder=FOLDER,
)
connectivity_data.ai_map = ai_map
connectivity_data.leafs = leafs
connectivity_data.get_injection_hemisphere_ids()
connectivity_data.align()
connectivity_data.get_centroids()
connectivity_data.get_data_matrices(default_structure_ids)
connectivity_data.get_crelines(data_info)
connectivity_data.get_summarystructures(data_info)

sid0 = list(connectivity_data.structure_datas.keys())[0]
eid0 = list(connectivity_data.structure_datas[sid0].experiment_datas.keys())[0]
contra_targetkey = connectivity_data.structure_datas[sid0].projection_mask.get_key(
    structure_ids=ontological_order_leaves, hemisphere_id=1
)
ipsi_targetkey = connectivity_data.structure_datas[sid0].projection_mask.get_key(
    structure_ids=ontological_order_leaves, hemisphere_id=2
)
# why is this ontological_order and not ontological_order_leavs
connectivity_data.get_regionalized_normalized_data(
    ontological_order, ipsi_targetkey, contra_targetkey
)

above_thresh = {}
thresh = 1
for m in range(12):
    sid = major_structure_ids[m]
    above_thresh[sid] = np.where(
        connectivity_data.structure_datas[sid].reg_inj.sum(axis=1) > thresh
    )[0]

for sid in major_structure_ids:
    connectivity_data.structure_datas[
        sid
    ].reg_proj_injnorm = connectivity_data.structure_datas[sid].reg_proj_injnorm[
        above_thresh[sid]
    ]
    connectivity_data.structure_datas[
        sid
    ].centroids = connectivity_data.structure_datas[sid].centroids[above_thresh[sid]]
    connectivity_data.creline[sid] = connectivity_data.creline[sid][above_thresh[sid]]
    connectivity_data.leafs[sid] = connectivity_data.leafs[sid][above_thresh[sid]]

reg_proj_injnorm = {
    sid: connectivity_data.structure_datas[sid].reg_proj_injnorm
    for sid in major_structure_ids
}

crelines = {sid: connectivity_data.creline[sid] for sid in major_structure_ids}
leaf2_index_matrices = get_indices_2ormore(leafs)
connectivity_data.get_creleaf_combos()
creleaf2_index_matrices = get_indices_2ormore(connectivity_data.creleaf_combos)
cre2_index_matrices = get_indices_2ormore(crelines)
creleaf2_evalindices = get_eval_indices(creleaf2_index_matrices)
cre2_index_matrices = {sid: cre2_index_matrices[sid] for sid in major_structure_ids}
major2_index_matrices = {
    sid: np.expand_dims(np.ones(crelines[sid].shape), 0) for sid in major_structure_ids
}
wtmajor2_index_matrices = {}
for sid in major_structure_ids:
    wtm = np.zeros(major2_index_matrices[sid].shape)
    wtmajor2_index_matrices[sid] = wtm[:, np.where(crelines[sid] == "C57BL/6J")[0]] = 1

ntotal = [
    connectivity_data.structure_datas[sid].reg_proj_injnorm.shape[0]
    for sid in major_structure_ids
]
ncreleaf2 = [len(creleaf2_evalindices[sid]) for sid in major_structure_ids]

df = pd.DataFrame(
    [ntotal, ncreleaf2],
    dtype=int,
    index=["Total", "Cre-Leaf"],
    columns=major_structures[reorder],
).transpose()
print(df.to_latex())

# Generate surfaces for weighting positional and Cre information
frac_learn = np.ones(12)
for m in range(12):
    sid = major_structure_ids[m]
    connectivity_data.structure_datas[
        sid
    ].loss_surface_cv_leaf = get_loss_surface_cv_spline(
        connectivity_data.structure_datas[sid].reg_proj_injnorm,
        connectivity_data.structure_datas[sid].centroids,
        connectivity_data.creline[sid],
        connectivity_data.leafs[sid],
        frac_learn[m],
    )
    connectivity_data.structure_datas[sid].smoothed_losses_leaf = get_embedding_cv(
        surface=connectivity_data.structure_datas[sid].loss_surface_cv_leaf,
        dists=pairwise_distances(connectivity_data.structure_datas[sid].centroids) ** 2,
        cre_distances_cv=connectivity_data.structure_datas[
            sid
        ].loss_surface_cv_leaf.cre_distances_cv,
    )

# Evaluate the Nadaraya-Watson models
gammas = np.asarray([0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000])

distances = {
    sid: pairwise_distances(connectivity_data.structure_datas[sid].centroids) ** 2
    for sid in major_structure_ids
}
print("Entering cross validation")
nw_creleaf_creleaf2 = CrossvalNW(
    reg_proj_injnorm, distances, creleaf2_index_matrices, creleaf2_evalindices, gammas
)
nw_creleaf_creleaf2.gammas = gammas
nw_creleaf_creleaf2.predictions = nw_creleaf_creleaf2.get_predictions()
nw_creleaf_creleaf2.get_results_loocv()
nw_creleaf_creleaf2.get_results_weightedloocv(leafs, crelines, ia_map)

nw_leaf_creleaf2 = CrossvalNW(
    reg_proj_injnorm, distances, leaf2_index_matrices, creleaf2_evalindices, gammas
)
nw_leaf_creleaf2.gammas = gammas
nw_leaf_creleaf2.predictions = nw_leaf_creleaf2.get_predictions()
nw_leaf_creleaf2.get_results_loocv()
nw_leaf_creleaf2.get_results_weightedloocv(leafs, crelines, ia_map)

nw_cremajor_creleaf2 = CrossvalNW(
    reg_proj_injnorm, distances, cre2_index_matrices, creleaf2_evalindices, gammas
)
nw_cremajor_creleaf2.gammas = gammas
nw_cremajor_creleaf2.predictions = nw_cremajor_creleaf2.get_predictions()
nw_cremajor_creleaf2.get_results_loocv()
nw_cremajor_creleaf2.get_results_weightedloocv(leafs, crelines, ia_map)

mean_gammas = np.ones(12) * 0.000001
mean_creleaf_creleaf2 = CrossvalNW(
    reg_proj_injnorm,
    distances,
    creleaf2_index_matrices,
    creleaf2_evalindices,
    mean_gammas,
)
mean_creleaf_creleaf2.gammas = mean_gammas  # ds
mean_creleaf_creleaf2.predictions = mean_creleaf_creleaf2.get_predictions()
mean_creleaf_creleaf2.get_results_loocv()
mean_creleaf_creleaf2.get_results_weightedloocv(
    connectivity_data.leafs, crelines, ia_map
)

nw_major_creleaf2 = CrossvalNW(
    reg_proj_injnorm, distances, major2_index_matrices, creleaf2_evalindices, gammas
)
nw_major_creleaf2.gammas = gammas
nw_major_creleaf2.predictions = nw_major_creleaf2.get_predictions()
nw_major_creleaf2.get_results_loocv()
nw_major_creleaf2.get_results_weightedloocv(leafs, crelines, ia_map)

nw_majorwt_creleaf2 = CrossvalNW(
    reg_proj_injnorm, distances, wtmajor2_index_matrices, creleaf2_evalindices, gammas
)
nw_majorwt_creleaf2.gammas = gammas
nw_majorwt_creleaf2.predictions = nw_major_creleaf2.get_predictions()
nw_majorwt_creleaf2.get_results_loocv()
nw_majorwt_creleaf2.get_results_weightedloocv(leafs, crelines, ia_map)

distances = {
    sid: connectivity_data.structure_datas[sid].smoothed_losses_leaf
    for sid in major_structure_ids
}
twostage_leaf_creleaf2 = CrossvalNW(
    reg_proj_injnorm, distances, leaf2_index_matrices, creleaf2_evalindices, gammas
)
twostage_leaf_creleaf2.gammas = gammas  # ds
twostage_leaf_creleaf2.predictions = twostage_leaf_creleaf2.get_predictions()
twostage_leaf_creleaf2.get_results_loocv()
twostage_leaf_creleaf2.get_results_weightedloocv(leafs, crelines, ia_map)

models = np.asarray(["Mean", "NW", "NW", "NW", "NW", "NW", "EL"])
datas = np.asarray(
    [
        r"$I_c \cap I_L$",
        r"$I_c \cap I_M$",
        r"$I_c \cap I_L$",
        r"$I_L$",
        r"$I_{wt} \cap I_M$",
        r"$I_M$",
        r"$I_L$",
    ]
)
multi_ind = np.vstack([models, datas])
multi_ind = np.asarray(multi_ind, dtype=str)
multi_ind = list(zip(*multi_ind))
multi_ind = pd.MultiIndex.from_tuples(
    multi_ind, names=[r"$\widehat f$", r"$\mathcal D$"]
)

results = pd.DataFrame(
    [
        mean_creleaf_creleaf2.meanloss_weighted,
        nw_cremajor_creleaf2.meanloss_weighted,
        nw_creleaf_creleaf2.meanloss_weighted,
        nw_leaf_creleaf2.meanloss_weighted,
        nw_majorwt_creleaf2.meanloss_weighted,
        nw_major_creleaf2.meanloss_weighted,
        twostage_leaf_creleaf2.meanloss_weighted,
    ],
    index=multi_ind,
    columns=major_structures,
).transpose()
trunc = lambda x: math.trunc(1000 * x) / 1000

results = results.applymap(trunc)

print(results.iloc[reorder].to_latex(index=True, escape=False))

fontsizes = {}
fontsizes[512] = 50
fontsizes[703] = 50
fontsizes[1089] = 50
fontsizes[1097] = 50
fontsizes[315] = 20
fontsizes[313] = 50
fontsizes[354] = 50
fontsizes[698] = 50
fontsizes[771] = 50
fontsizes[803] = 50
fontsizes[477] = 50
fontsizes[549] = 50
for sid in major_structure_ids:
    fig = plot_loss(twostage_leaf_creleaf2.weighted_losses[sid], fontsizes[sid])
    fig.savefig(
        "paper/KoelleConn_revision/figs/lossdetails_injnorm_cutoffremoved_" + str(sid),
        pad_inches=0,
    )

plot_loss_surface(
    connectivity_data.structure_datas[major_structure_ids[4]].loss_surface_cv_leaf
)

plot_loss_scatter(connectivity_data.structure_datas[315].loss_surface_cv_leaf)
plt.savefig("paper/KoelleConn_revision/figs/isocortexsurface_injnorm_cutoffremoved")

surfaces = {}
for m in range(len(major_structure_ids)):
    sid = major_structure_ids[m]
    surfaces[sid] = connectivity_data.structure_datas[sid].loss_surface_cv_leaf
    surfaces[sid].gamma = twostage_leaf_creleaf2.bestgamma_weighted[m]

with open(
    "analyses/results/EL_leafsurface_injnorm_cutoffremoved_060622_leafleaf.pickle", "wb"
) as handle:
    pickle.dump(surfaces, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(
    "analyses/results/EL_model_060622_injnorm_cutoffremoved_leafleaf.pickle", "wb"
) as handle:
    pickle.dump(twostage_leaf_creleaf2, handle, protocol=pickle.HIGEST_PROTOCOL)