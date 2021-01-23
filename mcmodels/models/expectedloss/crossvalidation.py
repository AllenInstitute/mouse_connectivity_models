import numpy as np
from .shapeconstrained import  get_surface_from_distances_spline
from .utils import  get_means
#from mcmodels.models.expectedloss.shapeconstrained import get_surface_from_distances_spline
#from mcmodels.models.expectedloss.utils import get_means

def get_embedding_cv(surface, dists, cre_distances_cv):
    ntrain = dists.shape[0]
    norms = surface.norms
    norms[np.where(norms == 0.)] = 1.
    leaf_pairs = np.asarray(np.where(~np.isnan(cre_distances_cv))).transpose()
    nlp = leaf_pairs.shape[0]

    losses = np.zeros((ntrain, ntrain))
    for i in range(nlp):
        d_ij = dists[leaf_pairs[i][0]][leaf_pairs[i][1]] / norms[0]
        p_ij = cre_distances_cv[leaf_pairs[i][0]][leaf_pairs[i][1]] / norms[1]
        losses[leaf_pairs[i][0]][leaf_pairs[i][1]] = surface.predict(np.asarray([[d_ij, p_ij]]))

    losses[np.where(losses == 0)] = np.nan

    return (losses)

def get_cre_distances_cv(proj, means_cast, sids, cres):
    nsamp = cres.shape[0]
    credist = np.empty((nsamp, nsamp))
    credist[:] = np.nan
    for i in range(nsamp):
        # print(i)
        meani = means_cast[sids[i]][cres[i]]
        ls = np.where(sids == sids[i])[0]  # np.where(leafs[sid] == leafs[sid][i])[0]
        crs = np.where(cres == cres[i])[0]
        ncr = len(np.intersect1d(ls, crs))

        meanloocvi = meani
        if ncr > 1:

            meanloocvi = (ncr * meani) / (ncr - 1) - (1 / (ncr - 1)) * proj[i]

            # this is wrong
            # meanloocvi = (ncr * meani ) / (ncr - 1) -   (1/ (ncr))* proj[i]
        else:
            meanloocvi = np.zeros(proj[i].shape[0])
            meanloocvi[:] = np.nan

        for j in range(nsamp):
            meanj = means_cast[sids[j]][cres[j]]
            if sids[j] == sids[i]:
                if cres[i] != cres[j]:
                    credist[j, i] = np.linalg.norm(meanloocvi - meanj)
                else:
                    if i != j:
                        credist[j, i] = 0.
    return (credist)


def get_loss_surface_cv_spline(projections, centroids, cres, sids, fraction):
    means_cast = get_means(projections, cres, sids)
    cre_distances_cv = get_cre_distances_cv(projections, means_cast, sids, cres)
    surface = get_surface_from_distances_spline(projections, centroids, cre_distances_cv, fraction)
    surface.cre_distances_cv = cre_distances_cv
    return (surface)
