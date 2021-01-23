import numpy as np
from mcmodels.models.crossvalidation import combine_predictions

def get_nwloocv_predictions_singlemodel_dists(projections, dists, gamma, model_indices, eval_indices):

    eval_index_val = np.where(eval_indices == 1)[0]
    model_index_val = np.where(model_indices == 1)[0]
    #print('e', eval_index_val, 'm', model_index_val)
    projections = np.asarray(projections, dtype=np.float32)
    nmod_ind = len(model_index_val)
    neval = len(eval_index_val)
    # nexp = centroids.shape[0]
    predictions = np.empty(projections.shape)
    predictions[:] = np.nan
    if len(model_index_val) > 0 and len(eval_index_val) > 0:
        # weights = np.exp(-dists[model_index_val][:, eval_index_val] / gamma)#np.exp(-dists[model_index_val] / gamma) #get_weights(centroids, gamma)
        for i in range(neval):
            matchindex = np.where(model_index_val == eval_index_val[i])[0]
            otherindices = np.setdiff1d(np.asarray(list(range(nmod_ind))), matchindex)
            # this order of operations is the fastest I found
            dists_i = dists[model_index_val][:, eval_index_val[i]] - np.min(
                dists[model_index_val[otherindices]][:, eval_index_val[i]])
            weights_i = np.exp(-dists_i * gamma)  # weights[i,:] / np.nansum(weights[i,:][otherindices])
            # print(np.nansum(weights[:,i][otherindices]))
            weights_i[matchindex] = 0
            weights_i = np.asarray(weights_i, dtype=np.float32)
            weights_i = weights_i / np.sum(weights_i)
            # weights_i[np.isnan(weights_i)] = 0.
            pred = np.dot(weights_i, projections[model_index_val])
            predictions[eval_index_val[i]] = pred

    return (predictions)

def get_nwloocv_predictions_multimodel_merge_dists(projections, dists, gammas, model_index_matrix, eval_index_matrix):
    predictions_unmerged = get_nwloocv_predictions_multimodel_dists(projections, dists, gammas, model_index_matrix,
                                                                    eval_index_matrix)
    #print(predictions_unmerged.shape)
    predictions_merged = combine_predictions(predictions_unmerged, eval_index_matrix)
    return (predictions_merged)

def get_nwloocv_predictions_multimodel_dists(projections, dists, gammas, model_index_matrix, eval_index_matrix):
    ntargets = projections.shape[1]
    nexp = projections.shape[0]
    nmodels = model_index_matrix.shape[0]
    ngammas = len(gammas)

    projections = np.asarray(projections, dtype=np.float32)
    predictions = np.empty((nmodels, ngammas, nexp, ntargets))
    predictions[:] = np.nan
    for m in range(nmodels):
        # print()
        predictions[m] = np.asarray([get_nwloocv_predictions_singlemodel_dists(projections, dists, gammas[g],
                                                                               model_index_matrix[m],
                                                                               eval_index_matrix[m]) for g in
                                     range(ngammas)])
        #print('m', m, len(np.where(model_index_matrix[m] == 1)[0]), np.nanmean(projections), np.nanmean(predictions[m]))

    return (predictions)

