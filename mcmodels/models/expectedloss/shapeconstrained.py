import numpy as np
#from mcmodels.models.expectedloss.utils import get_means
#from mcmodels.models.expectedloss.crossvalidation import get_cre_distances_cv
import random
import math
from pygam import LinearGAM,s


def get_surface_from_distances_spline(projections, centroids, cre_distances, fraction):
    nsamp = centroids.shape[0]
    pairs = np.asarray(
        np.where(~np.isnan(cre_distances))).transpose()  # not all cres will have distances, e.g. if not in same leaf
    ngp = pairs.shape[0]

    coordinates = np.zeros((ngp, 2))
    projection_distances = np.zeros((ngp, 1))
    for i in range(ngp):
        coordinates[i, 0] = np.linalg.norm(centroids[pairs[i][0]] - centroids[pairs[i][1]]) ** 2
        coordinates[i, 1] = cre_distances[pairs[i][0]][pairs[i][1]]
        projection_distances[i] = np.linalg.norm(projections[pairs[i][0]] - projections[pairs[i][1]]) ** 2
    coordinates_normed = coordinates / np.linalg.norm(coordinates, axis=0)  # **2
    coordinates_normed[np.where(np.isnan(coordinates_normed))] = 0.
    # was n_splines = 10
    surface = LinearGAM(s(0, constraints=['monotonic_inc', 'concave'], n_splines=10) +
                        s(1, constraints=['monotonic_inc', 'concave'], n_splines=10))

    # surface = NadarayaWatson(kernel='rbf',  gamma  = gamma)
    randos = random.sample(list(range(ngp)), math.floor(ngp * fraction))
    surface.fit(coordinates_normed[randos], projection_distances[randos])
    surface.coordinates_normed = coordinates_normed
    surface.norms = np.linalg.norm(coordinates, axis=0)
    surface.projection_distances = projection_distances
    return (surface)


def get_cre_distances(projections, means_cast, sids, cres):
    nsamp = cres.shape[0]
    credist = np.empty((nsamp, nsamp))
    credist[:] = np.nan
    for i in range(nsamp):
        # print(i)
        meani = means_cast[sids[i]][cres[i]]  # means_cast.loc[tuple([cres[i], sids[i]])]
        for j in range(nsamp):
            meanj = means_cast[sids[j]][cres[j]]
            if sids[j] == sids[i]:
                credist[i, j] = np.linalg.norm(meani - meanj) ** 2  # **2
    return (credist)

