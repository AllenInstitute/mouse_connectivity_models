import numpy as np
from mcmodels.models.expectedloss.utils import get_means
from mcmodels.models.expectedloss.shapeconstrained import get_embedding
from mcmodels.core import Mask
from sklearn.metrics import pairwise_distances
from mcmodels.models.voxel.utils import get_nw_predictions

def get_region_prediction3(cache, structure_data, prediction_region, prediction_sid_surface, experiment_sids_surface,
                           prediction_sid_nw, experiment_sids_nw, cre, gamma, surface):
    '''
    prediction_sid_surface: what is the sid for the surface (i.e. computing cre-means)
    experiment_sids_surface: what are the sids for the experiments w/ above
    prediction_sid_nw: what is the sid for Nadaraya-Watson
    experiment_sids_nw: what are the sids for the experiments w/ above
    '''

    nw_exp_ind = np.where(experiment_sids_nw == prediction_sid_nw)[0]
    surface_exp_ind = np.where(experiment_sids_surface == prediction_sid_surface)[0]

    nexp_surface = len(surface_exp_ind)
    cres_surface = structure_data.crelines[surface_exp_ind]

    mask = Mask.from_cache(cache, structure_ids=[prediction_region], hemisphere_id=2)

    projections = structure_data.reg_proj_norm[nw_exp_ind]
    centroids = structure_data.centroids[nw_exp_ind]

    projections_surface = structure_data.reg_proj_norm[surface_exp_ind]
    means = get_means(projections_surface, cres_surface, np.repeat(prediction_sid_surface, nexp_surface))

    if centroids.shape[0] > 0:

        if np.isin(cre, np.asarray(list(means[prediction_sid_surface].keys()))):
            losses = get_embedding(surface, pairwise_distances(centroids, mask.coordinates) ** 2, cres_surface, cre,
                                   means[prediction_sid_surface])
            predictions = get_nw_predictions(projections, losses, gamma)
            output = np.mean(predictions, axis=0)
            #output = output / np.sum(output)#/ np.linalg.norm(output)
        else:
            output = np.zeros(projections.shape[1])
            output[:] = np.nan
    else:
        output = np.zeros(projections.shape[1])
        output[:] = np.nan

    return (output)


def get_connectivity_matrices3(connectivity_data, surfaces, experiment_sids_surfaces, experiment_sids_nws,
                               model_ordering, source_ordering_surface, source_ordering_nw, source_ordering,
                               target_ordering, eval_cres):
    nsource = len(source_ordering)
    ncre = len(eval_cres)
    cache = connectivity_data.cache
    ipsi_target_regions = connectivity_data.ipsi_target_regions
    contra_target_regions = connectivity_data.contra_target_regions
    ipsi_indices = np.asarray([])
    contra_indices = np.asarray([])
    for iy in target_ordering:
        ipsi_indices = np.concatenate([ipsi_indices, np.where(ipsi_target_regions == iy)[0]])
        contra_indices = np.concatenate([contra_indices, np.where(contra_target_regions == iy)[0]])
    ipsi_indices = np.asarray(ipsi_indices, dtype=int)
    contra_indices = np.asarray(contra_indices, dtype=int)
    reorder = np.concatenate([ipsi_indices, len(ipsi_indices) + contra_indices])
    ntarget = len(reorder)

    connectivity = np.zeros((ncre, nsource, ntarget))
    connectivity[:] = np.nan
    # structure_major_dictionary = connectivity_data.structure_major_dictionary
    for c in range(ncre):
        for i in range(nsource):
            print(i, source_ordering[i])
            sid = int(model_ordering[i])  # structure_major_dictionary[source_ordering[i]]

            structure_data = connectivity_data.structure_datas[sid]
            # prediction_sid_surface = source_ordering_surface[i]
            experiment_sids_surface = experiment_sids_surfaces[sid]
            # prediction_sid_nw = source_ordering_nw[i]
            prediction_sid_nw = int(source_ordering_nw[i])

            prediction_sid_surface = int(source_ordering_surface[i])

            experiment_sids_nw = experiment_sids_nws[sid]
            prediction_region = source_ordering[i]
            cre = eval_cres[c]
            gamma = surfaces[sid].gamma
            surface = surfaces[sid]
            # print(prediction_sid_nw[0], prediction_sid_surface[0], 'yes')
            connectivity[c, i] = get_region_prediction3(cache,
                                                        structure_data,
                                                        prediction_sid_surface=prediction_sid_surface,
                                                        experiment_sids_surface=experiment_sids_surface,
                                                        prediction_sid_nw=prediction_sid_nw,
                                                        experiment_sids_nw=experiment_sids_nw,
                                                        prediction_region=prediction_region,
                                                        cre=cre,
                                                        gamma=gamma,
                                                        surface=surface)

    connectivity = connectivity[:, :, reorder]

    return (connectivity)
