import numpy as np


def get_nw_predictions(projections, dists, gamma):
    projections = np.asarray(projections, dtype=np.float32)
    neval = dists.shape[1]
    predictions = np.zeros((neval, projections.shape[1]))
    predictions[:] = np.nan

    for i in range(neval):
        dists_i = dists[:, i] - np.min(dists[:, i])
        weights_i = np.exp(-dists_i * gamma)
        weights_i = np.asarray(weights_i, dtype=np.float32)
        weights_i[np.isnan(weights_i)] = 0.0
        weights_i = weights_i / np.sum(weights_i)
        pred = np.dot(weights_i, projections)
        predictions[
            i
        ] = pred  # / np.sum(pred)#/ np.linalg.norm(pred)  # np.sum(pred)#np.linalg.norm(pred)

    return predictions
