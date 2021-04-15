import numpy as np

def get_nw_predictions(projections, dists, gamma):
    projections = np.asarray(projections, dtype=np.float32)
    neval = dists.shape[1]
    # nexp = centroids.shape[0]
    predictions = np.zeros((neval, projections.shape[1]))
    predictions[:] = np.nan

    # print(model_index_val.shape, eval_index_val.shape)
    # weights = np.exp(- dists / gamma)#np.exp(-dists[model_index_val] / gamma) #get_weights(centroids, gamma)
    for i in range(neval):
        dists_i = dists[:, i] - np.min(dists[:, i])
        # dists_i = dists[i,:] - np.min(dists[i,:])
        weights_i = np.exp(- dists_i * gamma)
        weights_i = np.asarray(weights_i, dtype=np.float32)
        weights_i[np.isnan(weights_i)] = 0.
        weights_i = weights_i / np.sum(weights_i)
        # predictions[i] = np.dot(weights_i, projections)
        pred = np.dot(weights_i, projections)
        predictions[i] = pred #/ np.sum(pred)#/ np.linalg.norm(pred)  # np.sum(pred)#np.linalg.norm(pred)

    return (predictions)
