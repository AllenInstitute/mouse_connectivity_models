import numpy as np
import itertools
import pandas as pd
from scipy.stats import kendalltau
from mcmodels.models.crossvalidation import (
    combine_predictions,
    get_best_hyperparameters,
    get_loss_best_hyp,
    get_loss,
)


def get_nwloocv_predictions_singlemodel_dists(
    projections, dists, gamma, model_indices, eval_indices
):
    eval_index_val = np.where(eval_indices == 1)[0]
    model_index_val = np.where(model_indices == 1)[0]
    projections = np.asarray(projections, dtype=np.float32)
    nmod_ind = len(model_index_val)
    neval = len(eval_index_val)
    predictions = np.empty(projections.shape)
    predictions[:] = np.nan
    if len(model_index_val) > 0 and len(eval_index_val) > 0:
        for i in range(neval):
            matchindex = np.where(model_index_val == eval_index_val[i])[0]
            otherindices = np.setdiff1d(np.asarray(list(range(nmod_ind))), matchindex)
            # this order of operations is the fastest I found
            dists_i = dists[model_index_val][:, eval_index_val[i]] - np.min(
                dists[model_index_val[otherindices]][:, eval_index_val[i]]
            )
            weights_i = np.exp(-dists_i * gamma)
            weights_i[matchindex] = 0
            weights_i = np.asarray(weights_i, dtype=np.float32)
            weights_i = weights_i / np.sum(weights_i)
            pred = np.dot(weights_i, projections[model_index_val])
            predictions[eval_index_val[i]] = pred

    return predictions


def get_nwloocv_predictions_multimodel_merge_dists(
    projections, dists, gammas, model_index_matrix, eval_index_matrix
):
    predictions_unmerged = get_nwloocv_predictions_multimodel_dists(
        projections, dists, gammas, model_index_matrix, eval_index_matrix
    )
    predictions_merged = combine_predictions(predictions_unmerged, eval_index_matrix)
    return predictions_merged


def get_nwloocv_predictions_multimodel_dists(
    projections, dists, gammas, model_index_matrix, eval_index_matrix
):
    ntargets = projections.shape[1]
    nexp = projections.shape[0]
    nmodels = model_index_matrix.shape[0]
    ngammas = len(gammas)

    projections = np.asarray(projections, dtype=np.float32)
    predictions = np.empty((nmodels, ngammas, nexp, ntargets))
    predictions[:] = np.nan
    for m in range(nmodels):
        # print()
        predictions[m] = np.asarray(
            [
                get_nwloocv_predictions_singlemodel_dists(
                    projections,
                    dists,
                    gammas[g],
                    model_index_matrix[m],
                    eval_index_matrix[m],
                )
                for g in range(ngammas)
            ]
        )

    return predictions


class CrossvalNW:
    def __init__(self, data, distances, model_indices, eval_indices, gammas):
        self.data = data
        self.distances = distances
        self.model_indices = model_indices
        self.models = np.asarray(list(model_indices.keys()))
        self.eval_indices = eval_indices
        self.gammas = gammas

    def get_predictions(self):
        model_indices = self.model_indices
        gammas = self.gammas
        data = self.data
        distances = self.distances
        models = self.models

        predictions = {}
        for sid in models:
            predictions[sid] = get_nwloocv_predictions_multimodel_merge_dists(
                data[sid],
                distances[sid],
                gammas,
                model_indices[sid],
                model_indices[sid],
            )

        return predictions

    def get_results_loocv(self):
        data = self.data
        predictions = self.predictions
        eval_indices = self.eval_indices
        gammas = self.gammas
        a = [list(range(len(gammas)))]
        keys = np.asarray(list(itertools.product(*a)))

        losses = get_loss(data, predictions, eval_indices, eval_indices, keys=keys)
        bestgamma = get_best_hyperparameters(losses, keys)
        meanloss = get_loss_best_hyp(losses, bestgamma)

        self.losses = losses
        self.bestgamma = gammas[bestgamma]
        self.meanloss = meanloss

    def get_results_weightedloocv(self, structures, crelines, ia_map):
        eval_indices = self.eval_indices
        losses = self.losses
        gammas = self.gammas
        models = self.models
        ngamma = len(gammas)
        nmodels = len(models)
        nw_losses_all = np.zeros((ngamma, nmodels))
        meanlosses = {}
        for g in range(ngamma):
            for m in range(nmodels):
                sid = models[m]
                leaf_sid = np.asarray(
                    [ia_map[structures[sid][i]] for i in range(len(structures[sid]))]
                )[eval_indices[sid]]
                comboloss = pd.DataFrame(
                    np.asarray(
                        [
                            losses[sid][[g], :][0],
                            crelines[sid][eval_indices[sid]],
                            leaf_sid,
                        ]
                    ).transpose()
                )

                comboloss.columns = np.asarray(["NW-CreSum-Loss", "Cre", "Sum"])
                #        comboloss['EL-Sum-Loss'] = pd.to_numeric(comboloss['EL-Sum-Loss'])
                comboloss["NW-CreSum-Loss"] = pd.to_numeric(comboloss["NW-CreSum-Loss"])

                meanlosses[sid] = comboloss.pivot_table(
                    values="NW-CreSum-Loss", index="Cre", columns="Sum", aggfunc=np.mean
                )
                nw_losses_all[g, m] = np.nanmean(np.asarray(meanlosses[sid]))

        self.weighted_losses = meanlosses
        self.bestgamma_weighted = gammas[np.argmin(nw_losses_all, axis=0)]
        self.meanloss_weighted = np.min(nw_losses_all, axis=0)
