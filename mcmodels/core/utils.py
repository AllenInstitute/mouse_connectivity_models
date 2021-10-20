"""
Module containing utility functions for the :mod:`mcmodels.core` module.
"""
# Authors: Joseph Knox <josephk@alleninstitute.org>
# License: Allen Institute Software License

from __future__ import division

import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import pairwise_kernels
from mcmodels.utils import nonzero_unique, unionize
from mcmodels.core import Mask
from collections import Counter
#from mcmodels.core import ExperimentData
from mcmodels.core.experiment_data import ExperimentData





def get_ccf_data(cache, experiment_id, folder):

    eid_data = ExperimentData(experiment_id)
    eid_data.data_quality_mask = cache.get_data_mask(experiment_id, folder + 'experiment_'+ str(experiment_id) + '/data_mask_100.nrrd')[0]
    eid_data.injection_signal = cache.get_injection_density(experiment_id, folder + 'experiment_'+ str(experiment_id) + '/injection_density_100.nrrd')[0]
    eid_data.injection_fraction = cache.get_injection_fraction(experiment_id, folder + 'experiment_'+ str(experiment_id) + '/injection_fraction_100.nrrd')[0]
    eid_data.projection_signal = cache.get_projection_density(experiment_id, folder+ 'experiment_'+ str(experiment_id) + '/projection_density_100.nrrd')[0]
    return(eid_data)

def get_centroid(density):
    """Computes centroid in index coordinates.

    Parameters
    ----------
    injection_density : array, shape (x_ccf, y_ccf, z_ccf)
        injection_density data volume.

    Returns
    -------
        centroid of injection_density in index coordinates.
    """
    nnz = density.nonzero()
    coords = np.vstack(nnz)

    return np.dot(coords, density[nnz]) / density.sum()


def get_injection_hemisphere_id(injection_density, majority=False):
    """Gets injection hemisphere based on injection density.

    Defines injection hemisphere by the ratio of the total injection_density
    in each hemisphere.

    Parameters
    ----------
    injection_density : array, shape (x_ccf, y_ccf, z_ccf)
        injection_density data volume.

    Returns
    -------
    int : in (1,2,3)
        injection_hemisphere
    """
    if injection_density.ndim != 3:
        raise ValueError("injection_density must be 3-array not (%d)-array"
                         % injection_density.ndim)

    # split along depth dimension (forces arr.shape[2] % 2 == 0)
    hemis = np.dsplit(injection_density, 2)
    hemi_sums = tuple(map(np.sum, hemis))

    # if not looking for either l or r
    if not majority and all(hemi_sums):
        return 3

    left_sum, right_sum = hemi_sums
    if left_sum > right_sum:
        return 1

    return 2


def get_indices(ids):
    ids_unique = np.unique(ids)
    output = np.zeros((len(ids_unique), len(ids)), dtype=int)
    for i in range(len(ids_unique)):
        output[i, np.where(ids == ids_unique[i])[0]] = 1
    return (output)

# get indices of firstlist in firstlisttest in categories defined by secondlist
def get_indices2(firstlist, firstlisttest, secondlist):
    sl_unique = np.unique(secondlist)
    output = np.zeros((len(sl_unique), len(secondlist)), dtype=int)
    for i in range(len(sl_unique)):
        output[i, np.intersect1d(np.where(np.isin(firstlist, firstlisttest))[0],
                                 np.where(secondlist == sl_unique[i])[0])] = 1
    return (output)

def get_indices_2ormore(index_dictionary):

    major_structure_ids = np.asarray(list(index_dictionary.keys()))
    indices_2ormore = {}
    for sid in major_structure_ids:
        indices = get_indices(index_dictionary[sid])  # eval_indices
        indices_2ormore[sid] = screen_index_matrices(indices,indices)
    return(indices_2ormore)


def screen_index_matrices(model_index_matrices, eval_index_matrices):
    # alter eval_indices to remove model index in cases where there is only one experiment in the model

    nmodels = model_index_matrices.shape[0]
    eval_index_matrices2 = eval_index_matrices.copy()
    for m in range(nmodels):
        eval_index_matrices2[m] = screen_indices(model_index_matrices[m], eval_index_matrices[m])

    return (eval_index_matrices2)


def screen_indices(model_indices, eval_indices):
    eval_indices2 = eval_indices.copy()
    mod_loc = np.where(model_indices == 1)[0]
    if len(mod_loc) == 1:
        eval_indices2[mod_loc] = 0
    return (eval_indices2)


def screen_index_matrices2(model_index_matrices, eval_index_matrices):
    # alter model and eval matrices to be nonzero only when there are at least two experiments in the model
    # it can be useful for when model_index_matrices is a subset of eval_index_matrices
    # nmodels = model_index_matrices.shape[0]
    include_per_model = model_index_matrices.sum(axis=1)
    to_exclude = np.where(include_per_model <= 1)[0]
    # to_include = np.where(include_per_model > 0)[0]

    model_index_matrices2 = model_index_matrices.copy()
    eval_index_matrices2 = eval_index_matrices.copy()
    model_index_matrices2[to_exclude] = 0
    eval_index_matrices2[to_exclude] = 0

    return (model_index_matrices2, eval_index_matrices2)


def screen_index_matrices3(model_index_matrices, eval_index_matrices):
    # alter model and eval matrices to be nonzero only when there are at least one experiments in the model
    # it can be useful for when model_index_matrices is a subset of eval_index_matrices
    # nmodels = model_index_matrices.shape[0]
    include_per_model = model_index_matrices.sum(axis=1)
    to_exclude = np.where(include_per_model < 1)[0]
    # to_include = np.where(include_per_model > 0)[0]

    model_index_matrices2 = model_index_matrices.copy()
    eval_index_matrices2 = eval_index_matrices.copy()
    model_index_matrices2[to_exclude] = 0
    eval_index_matrices2[to_exclude] = 0

    to_remove = np.where(include_per_model == 1)[0]
    eval_index_matrices2[to_remove] = 0

    return (model_index_matrices2, eval_index_matrices2)

def get_eval_indices(eval_index_matrices):
    eval_indices = {}
    major_structure_ids = np.asarray(list(eval_index_matrices.keys()))
    for sid in major_structure_ids:
        eval_indices[sid] = np.where(eval_index_matrices[sid].sum(axis = 0) > 0)[0]
    return(eval_indices)


def get_matched_index_matrices(model_index_matrices, eval_index_matrices):
    major_structure_ids = np.asarray(list(eval_index_matrices.keys()))
    matched_model_index_matrices = {}
    # eval_indices = get_eval_indices(eval_index_matrices)
    for sid in major_structure_ids:
        matched_model_index_matrices[sid] = np.zeros(eval_index_matrices[sid].shape)
        #         model_indices = []
        #         for m in range(model_index_matrices[sid].shape[0]):
        #             model_indices = model_indices.append(np.where(model_index_matrices[sid] == 1)[0])
        for m in range(eval_index_matrices[sid].shape[0]):
            for n in range(model_index_matrices[sid].shape[0]):
                model_indices = np.where(model_index_matrices[sid][n] == 1)[0]
                eval_indices = np.where(eval_index_matrices[sid][m] == 1)[0]
                if len(np.intersect1d(model_indices, eval_indices)) > 0:
                    # difference = model_index_matrices[n] - eval_index_matrices[m]
                    # if np.min(difference != -1):
                    print(model_index_matrices[sid][n].sum())
                    matched_model_index_matrices[sid][m] = model_index_matrices[sid][n]
        matched_model_index_matrices[sid] = np.asarray(matched_model_index_matrices[sid], dtype=int)
    # print(matched_model_index_matrices)
    return (matched_model_index_matrices)


def get_masked_data_volume(data_volume, data_mask, tolerance=0.0):
    """Masks a given data volume in place.

    Parameters
    ----------
    data_volume : array, shape (x_ccf, y_ccf, z_ccf)
        Data volume to be masked.

    data_mask : array, shape (x_ccf, y_ccf, z_ccf)
        data_mask for given experiment (values in [0,1])
        See allensdk.core.mouse_connectivity_cache for more info.

    tolerance : float, optional (default=0.0)
        tolerance with which to define bad voxels in data_mask.


    Returns
    -------
    data_volume
        data_volume parameter masked in place.

    """
    if data_volume.shape != data_mask.shape:
        raise ValueError("data_volume (%s) and data_mask (%s) must be the same "
                         "shape!" % (data_volume.shape, data_mask.shape))

    # mask data volume
    data_volume[data_mask < tolerance] = 0.0

    return data_volume



###############

def get_twoormore(classdict):
    major_structure_ids = np.asarray(list(classdict.keys()))
    output = {}
    for sid in major_structure_ids:
        count = Counter(classdict[sid])
        freq = np.asarray([count[classdict[sid][i]] for i in range(len(classdict[sid]))])
        output[sid] = np.where(freq > 1)[0]
    return (output)

#order = ontological_order
def get_regionalized_normalized_data(msvds, cache, source_order, ipsi_key, contra_key): #experiments_minor_structures):
    '''

    :param msvds: Class dictionary holding data
    :param cache: AllenSDK cache
    :param source_order: Source key (tautologically ipsilateral due to hemisphere mirroring)
    :param ipsi_key: Ipsilateral target key
    :param contra_key:  Contralateral target key
    :return: msvds: Class dictionary holding average data
    '''
    major_structure_ids = np.asarray(list(msvds.keys()))
    for sid in major_structure_ids:
        # print()
        msvd = msvds[sid]
        #nexp = msvd.projections.shape[0]

        #minor_structures = np.unique(experiments_minor_structures[sid])
        #nmins = len(minor_structures)

        projections = msvd.projections
        ipsi_proj = unionize(projections, ipsi_key)
        contra_proj = unionize(projections, contra_key)
        reg_proj = np.hstack([ipsi_proj, contra_proj])
        msvd.reg_proj = reg_proj

        ipsi_target_regions, ipsi_target_counts = nonzero_unique(ipsi_key, return_counts=True)
        contra_target_regions, contra_target_counts = nonzero_unique(contra_key, return_counts=True)
        target_counts = np.concatenate([ipsi_target_counts, contra_target_counts])
        reg_proj_vcount_norm = np.divide(reg_proj, target_counts[np.newaxis, :])
        msvd.reg_proj_vcount_norm = reg_proj_vcount_norm

        projections = msvds[sid].reg_proj_vcount_norm
        projections = projections / np.expand_dims(np.linalg.norm(projections, axis=1), 1)
        msvd.reg_proj_vcount_norm_renorm = projections

        source_mask = Mask.from_cache(cache, structure_ids=[sid], hemisphere_id=2)
        source_key = source_mask.get_key(structure_ids=source_order)
        source_regions, source_counts = nonzero_unique(source_key, return_counts=True)

        injections = msvd.injections
        reg_ipsi_inj = unionize(injections, source_key)
        msvd.reg_inj = reg_ipsi_inj
        reg_inj_vcount_norm = np.divide(reg_ipsi_inj, source_counts[np.newaxis, :])
        msvd.reg_inj_vcount_norm = reg_inj_vcount_norm
        #msvd.reg_proj_vcountnorm_totalnorm =

    return (msvds)


def get_ontological_order_leaf(leafs, ontological_order, st):
    major_structure_ids = np.asarray(list(leafs.keys()))
    leaf_present = np.concatenate([leafs[sid] for sid in major_structure_ids])
    ontological_order_leaf = np.asarray([])
    for i in range(len(ontological_order)):
        # which of these are in leafs
        stos = np.asarray(st.child_ids([ontological_order[i]]))
        which_stos = np.asarray(np.where(np.isin(stos, leaf_present)[0])[0], dtype=int)
        if len(which_stos) > 0:
            print(i)
            ontological_order_leaf = np.append(ontological_order_leaf, stos[0][which_stos])
        if np.isin(ontological_order[i], leaf_present):
            print(i)
            ontological_order_leaf = np.append(ontological_order_leaf, ontological_order[i])
    ontological_order_leaf = np.asarray(ontological_order_leaf, dtype=int)
    return (ontological_order_leaf)

def get_connectivity(msvds, cache, ia_map, hyperparameters, source_ordering, target_ordering, leafs, creline,
                     experiments_minor_structures, ipsi_key, contra_key):

    source_exp_countvec, source_exp_countvec_wt = get_countvec(source_ordering, ia_map, creline,
                                                               experiments_minor_structures)

    major_structure_ids = np.asarray(list(msvds.keys()))
    nms = len(major_structure_ids)
    prediction_union_norms = {}
    source_region_save = np.asarray([])

    for m in range(nms):
        sid = major_structure_ids[m]
        gamma = hyperparameters[m]
        minor_structures = source_ordering[np.where(np.isin(source_ordering, np.unique(leafs[sid])))]
        # ontological_order_leaf np.unique(leafs[sid]) # this should be in ontological order # np.unique(leafs[sid])#source_ordering[sid]#
        prediction_union_norms[m] = {}
        for n in range(len(minor_structures)):
            print(n)
            minor_structure_inds = np.where(leafs[sid] == minor_structures[n])[0]
            # meezy = minor_structures[n]
            im = Mask.from_cache(
                cache,
                structure_ids=[minor_structures[n]],
                hemisphere_id=2)
            weights = pairwise_kernels(X=msvds[sid].centroids[minor_structure_inds], Y=im.coordinates, metric='rbf',
                                       gamma=gamma, filter_params=True)
            weights = weights / weights.sum(axis=0)
            weights[np.where(np.isnan(weights))] = 0.
            predictions = np.dot(weights.transpose(), msvds[sid].reg_proj_vcount_norm_renorm[minor_structure_inds])

            # average over source region voxels
            union_key = im.get_key(structure_ids=source_ordering, hemisphere_id=2)
            source_regions, source_counts = nonzero_unique(union_key, return_counts=True)
            prediction_union = unionize(predictions.transpose(), union_key)
            prediction_union_norms[m][n] = prediction_union.transpose() / np.expand_dims(source_counts, 1)
            source_region_save = np.append(source_region_save, source_regions)

    prediction_union_norms_ms = {}
    for m in range(nms):
        prediction_union_norms_ms[m] = np.vstack(
            [prediction_union_norms[m][n] for n in range(len(prediction_union_norms[m].keys()))])

    cd = np.vstack([prediction_union_norms_ms[m] for m in range(len(prediction_union_norms_ms.keys()))])

    # get row names
    rownames = [ia_map[source_ordering[i]] for i in range(len(source_ordering))]
    # rownames = np.asarray(rownames)[np.where(source_exp_countvec !=0)[0]]

    # get column names
    ipsi_target_regions, ipsi_target_counts = nonzero_unique(ipsi_key, return_counts=True)
    contra_target_regions, contra_target_counts = nonzero_unique(contra_key, return_counts=True)
    target_order = lambda x: np.array(target_ordering)[np.isin(target_ordering, x)]
    permutation = lambda x: np.argsort(np.argsort(target_order(x)))
    targ_ids = np.concatenate([ipsi_target_regions[permutation(ipsi_target_regions)],
                               contra_target_regions[permutation(contra_target_regions)]])
    colnames = np.asarray([ia_map[targ_ids[i]] for i in range(len(targ_ids))])

    # reorder rows and columns
    targ_ords = np.concatenate(
        [permutation(ipsi_target_regions), len(ipsi_target_regions) + permutation(contra_target_regions)])
    row_reorder = np.asarray([])
    source_region_save = np.asarray(source_region_save, dtype=int)
    for i in range(len(source_ordering)):
        inx = np.where(source_region_save == int(source_ordering[i]))[0]
        if len(inx) > 0:
            row_reorder = np.append(row_reorder, inx)
    row_reorder = np.asarray(row_reorder, dtype=int)

    df = pd.DataFrame(cd[row_reorder][:, targ_ords], index=rownames, columns=np.asarray(colnames))
    return (df)

def get_wt_inds(creline):
    major_structure_ids = np.asarray(list(creline.keys()))
    wt_2ormore = {}
    for sid in major_structure_ids:
        wt_inds = np.where(creline[sid] == 'C57BL/6J')[0]
        wt_2ormore[sid] = np.asarray([])
        if len(wt_inds) > 1:
            wt_2ormore[sid] = np.append(wt_2ormore[sid], wt_inds)
        wt_2ormore[sid] = np.asarray(wt_2ormore[sid], dtype=int)
    return (wt_2ormore)


#indices was wt_2ormore[sid]
def get_nw_loocv(msvd, indices, loocv, hyperparameters):

    if len(indices) > 1:
        projections = msvd.reg_proj_vcount_norm_renorm
        centroids = msvd.centroids
        nreg = projections.shape[1]
        nexp = projections.shape[0]
        nhyp = hyperparameters.shape[0]
        loocv_predictions = np.zeros((nhyp, nexp, nreg))
        for g in range(nhyp):
            loocv_predictions[g, indices] = loocv(projections[indices], centroids[indices], hyperparameters[g])
        return (loocv_predictions)
    else:
        return (np.asarray([]))

#groups is better than indices
def get_nw_loocv2(msvd, loocv, hyperparameters, groups):

    if len(indices) > 1:
        projections = msvd.reg_proj_vcount_norm_renorm
        centroids = msvd.centroids
        nreg = projections.shape[1]
        nexp = projections.shape[0]
        nhyp = hyperparameters.shape[0]
        loocv_predictions = np.zeros((nhyp, nexp, nreg))
        for g in range(nhyp):
            loocv_predictions[g, indices] = loocv(projections[indices], centroids[indices], hyperparameters[g],groups)
        return (loocv_predictions)
    else:
        return (np.asarray([]))


def get_countvec(ontological_order, ia_map, creline, experiments_minor_structures):
    major_structure_ids = np.asarray(list(creline.keys()))
    sourcenames = np.asarray([ia_map[ontological_order[i]] for i in range(len(ontological_order))])
    source_exp_counts = {}
    source_exp_counts_wt = {}
    for i in range(len(sourcenames)):
        source_exp_counts[sourcenames[i]] = 0
        source_exp_counts_wt[sourcenames[i]] = 0
        for sid in major_structure_ids:
            source_exp_counts[sourcenames[i]] += len(np.where(experiments_minor_structures[sid] == sourcenames[i])[0])
            source_exp_counts_wt[sourcenames[i]] += len(
                np.intersect1d(np.where(experiments_minor_structures[sid] == sourcenames[i])[0],
                               np.where(creline[sid] == 'C57BL/6J')))
    source_exp_countvec = np.asarray(list(source_exp_counts.values()))
    source_exp_countvec_wt = np.asarray(list(source_exp_counts_wt.values()))
    return (source_exp_countvec, source_exp_countvec_wt)
#
# def get_leaves_ontologicalorder(msvd, ontological_order):
#     '''
#
#     :param msvd:
#     :param ontological_order:
#     :return: The leaf order associated with the 'ontological_order' of summary structures
#     '''
#     levs = msvd.experiments[list(msvd.experiments.keys())[0]].projection_mask.reference_space.structure_tree.child_ids(
#         ontological_order)
#     flat_list = np.asarray([item for sublist in levs for item in sublist])
#
#     nss = len(levs)
#     leavves = np.asarray([])
#     for i in range(nss):
#         if len(levs[i]) > 0:
#             leavves = np.append(leavves, levs[i])
#         else:
#             leavves = np.append(leavves, ontological_order[i])
#     return (leavves)


def get_leaves_ontologicalorder(connectivity_data, ontological_order):
    '''

    :param msvd:
    :param ontological_order:
    :return: The leaf order associated with the 'ontological_order' of summary structures
    '''
    sid0 = list(connectivity_data.structure_datas.keys())[0]
    #eid0 = list(connectivity_data.structure_datas[sid0].experiment_datas.keys())[0]
    levs = connectivity_data.structure_datas[sid0].projection_mask.reference_space.structure_tree.child_ids(
        ontological_order)
    flat_list = np.asarray([item for sublist in levs for item in sublist])

    nss = len(levs)
    leavves = np.asarray([])
    for i in range(nss):
        if len(levs[i]) > 0:
            leavves = np.append(leavves, levs[i])
        else:
            leavves = np.append(leavves, ontological_order[i])
    return (leavves)


def get_minorstructure_dictionary(msvds, data_info):
    experiments_minor_structures = {}
    major_structure_ids = np.asarray(list(msvds.keys()))
    for sid in major_structure_ids:
        eids = np.asarray(list(msvds[sid].experiments.keys()))
        experiments_minor_structures[sid] = get_minorstructures(eids, data_info)
    return (experiments_minor_structures)



def get_cre_status(data_info, msvds):
    major_structure_ids = np.asarray(list(msvds.keys()))
    exps = np.asarray(data_info.index.values , dtype = np.int)
    creline = {}
    for sid in major_structure_ids:
        msvd = msvds[sid]
        experiment_ids = np.asarray(list(msvd.experiments.keys()))
        nexp = len(experiment_ids)
        creline[sid] = np.zeros(nexp, dtype = object)
        for i in range(len(experiment_ids)):
            index = np.where(exps == experiment_ids[i])[0][0]
            creline[sid][i] = data_info['transgenic-line'].iloc[index]
    return(creline)

#
# def get_matrices(experiments):
#     get_data = lambda x: (get_centroid(x),
#                           get_injection(x,True),
#                           get_projection(x, True))
#     arrays = map(get_data, yield_experiments(experiments))
#     centroids, injections, projections = map(np.array, zip(*arrays))
#     return(centroids, injections, projections)
#
# def get_centroid(experiment):
#     """Returns experiment centroid"""
#     return experiment.centroid
#
# def get_injection(experiment, normalized_injection):
#     # print('ts',experiment.normalized_injection)
#     """Returns experiment injection masked & flattened"""
#     injection = experiment.get_injection(normalized_injection)
#     return experiment.injection_mask.mask_volume(injection)
#
# def get_projection(experiment, normalized_projection):
#     """Returns experiment projection masked & flattened"""
#     projection = experiment.get_projection(normalized_projection)
#     return experiment.projection_mask.mask_volume(projection)

def yield_experiments(experiments):
    ev = experiments.values()
    keys = np.asarray(list(experiments.keys()))
    for i in range(len(keys)):
        yield (experiments[keys[i]])

def get_loss_paper(y,yhat):
    loss = 2* np.linalg.norm(y - yhat)**2 / (np.linalg.norm(y)**2 + np.linalg.norm(yhat)**2)
    return(loss)


def get_structure_id(cache, acronym):
    try:
        return cache.get_structure_tree().get_structures_by_acronym([acronym])[0]['id']
    except KeyError:
        raise ValueError("structure acronym (%s) is not valid" % acronym)



# get columns with number values exceeding thresh
def get_reduced_matrix_ninj(X, thresh, number):
    X_thresh = X.copy()
    X_thresh[np.where(X > thresh)] = 1.
    X_thresh[np.where(X <= thresh)] = 0.
    n_inj = X_thresh.sum(axis=0)
    inds = np.where(n_inj >= number)[0]

    return (X[:, inds], inds)





def get_ordered_summary_structures(mcc,set_id = 687527945):
    # TODO : replace with json of wanted structures

    """Returns structure ids of summary structures - fiber tracts (and 934)"""
    ss_regions = mcc.get_structure_tree().get_structures_by_set_id([set_id])
    #ss_regions = mcc.get_structure_tree().get_structures_by_set_id([167587189])

    # 934 not in 100 micron!!!!! (dont want fiber tracts)
    ids, orders = [], []
    for region in ss_regions:
        if region["id"] not in [934, 1009]:
            ids.append(region["id"])
            orders.append(region["graph_order"])

    # return ids sorted by graph order
    ids = np.asarray(ids)
    return ids[np.argsort(orders)]


def get_minorstructures(eids, data_info):
    #eids = np.asarray(list(msvd.experiments.keys()))
    experiments_minors = np.zeros(len(eids), dtype=object)

    for i in range(len(eids)):
        experiment_id = eids[i]
        experiments_minors[i] = data_info['primary-injection-structure'].loc[experiment_id]

    return (experiments_minors)


####old

