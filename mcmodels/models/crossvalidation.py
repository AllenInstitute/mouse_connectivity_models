import numpy as np
from mcmodels.core.utils import get_loss_paper


class Crossval:
    """
    dummy class for cross validation
    """

    def __init__(self):
        2 + 2


def combine_predictions(predictions, eval_index_matrix):
    """

    :param predictions:
    :param eval_index_matrix:
    :return:
    """
    nmodels, ngammas, nexp, ntargets = predictions.shape
    combined_predictions = np.empty((ngammas, nexp, ntargets))
    for m in range(nmodels):
        combined_predictions[:, np.where(eval_index_matrix[m] == 1)[0]] = predictions[
            m
        ][:, np.where(eval_index_matrix[m] == 1)[0]]

    return combined_predictions


def get_best_hyperparameters(losses, keys):
    major_structure_ids = np.asarray(list(losses.keys()))
    nms = len(major_structure_ids)
    nkey = keys.shape[1]
    output = np.empty((nms, nkey))
    for m in range(nms):
        # print(m)
        sid = major_structure_ids[m]
        lvec = np.asarray([np.nanmean(losses[sid][key]) for key in keys])
        if np.any(~np.isnan(lvec)):
            output[m] = keys[np.nanargmin(lvec)]
        # if len(np.where(np.isnan(np.nanmean(losses[sid][:,:], axis = 1)))[0]) < losses[sid].shape[0]:
        #    output[m] = np.nanargmin(np.nanmean(losses[sid][:,:], axis = 1))

    output = np.asarray(output, dtype=int)
    return output


def get_loss(true_dict, prediction_dict, pred_ind=None, true_ind=None, keys=None):
    output = {}
    major_structure_ids = list(prediction_dict.keys())
    nms = len(major_structure_ids)
    ngam = prediction_dict[major_structure_ids[0]].shape[0]
    nalph = prediction_dict[major_structure_ids[0]].shape[1]
    for m in range(nms):
        sid = major_structure_ids[m]
        if pred_ind == None:
            # prediction_dict and true_dict will contain predictions for 'bad' experiments with no recorded injection
            # when we have the wild type predictions, the subset is what is good among the wild types
            # so 'true' subsetting is always good, since it is w.r.t. the full injection
            # but prediction needs good w.r.t. wt
            pind = np.asarray(list(range(prediction_dict[sid].shape[1])), dtype=int)
        else:
            pind = pred_ind[sid]
        if true_ind == None:
            tind = np.asarray(list(range(true_dict[sid].shape[0])), dtype=int)
        else:
            tind = true_ind[sid]

        nexp = len(pind)

        output[sid] = np.zeros(
            np.append([len(np.unique(keys[:, i])) for i in range(keys.shape[1])], nexp)
        )

        for j in range(keys.shape[0]):
            output[sid][tuple(keys[j])] = np.asarray(
                [
                    get_loss_paper(
                        true_dict[sid][tind[i]],
                        prediction_dict[sid][tuple(keys[j])][pind[i]],
                    )
                    for i in range(nexp)
                ]
            )

    return output


def get_loss_best_hyp(losses, hyps):
    major_structure_ids = np.asarray(list(losses.keys()))
    nms = len(major_structure_ids)
    output = np.zeros(nms)
    for m in range(nms):
        sid = major_structure_ids[m]
        output[m] = np.nanmean(losses[sid][hyps[m], :])
    return output
