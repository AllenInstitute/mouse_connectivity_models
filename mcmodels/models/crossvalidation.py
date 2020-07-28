import numpy as np
from mcmodels.models.homogeneous import svd_subset_selection, HomogeneousModel
from mcmodels.regressors.nonparametric.nadaraya_watson import get_weights
from mcmodels.core.utils import get_reduced_matrix_ninj, get_loss_paper
from sklearn import decomposition

def get_loocv_predictions(projections, centroids, gamma):
    projections = np.asarray(projections, dtype=np.float32)
    nexp = centroids.shape[0]
    predictions = projections.copy()
    weights = get_weights(centroids, gamma)

    for i in range(nexp):
        otherindices = np.setdiff1d(np.asarray(list(range(nexp))), i)
        weights_i = weights[i] / weights[i][otherindices].sum()
        weights_i[i] = 0
        weights_i = np.asarray(weights_i, dtype=np.float32)
        pred = np.dot(weights_i, projections)
        predictions[i] = pred

    return (predictions)


def get_loocv_predictions_nnlinear_ridge(projections, injections, lam):
    # projections = reg_proj_vcount_norm
    # injections = reg_inj_vcount_norm
    projections = np.asarray(projections, dtype=np.float32)
    injections = np.asarray(injections, dtype=np.float32)
    nexp = projections.shape[0]
    predictions = projections.copy()

    # homo_est = HomogeneousModel(kappa=kappa)

    for i in range(nexp):
        print(i)
        otherindices = np.setdiff1d(np.asarray(list(range(nexp))), i)
        d = injections[otherindices].shape[1]
        X = injections[otherindices].copy()
        # not normalizing for now
        # norms = np.linalg.norm(X, axis = 0)
        # X = X / norms
        # X[:,np.where(norms < 0.0000001)[0]] = 0.
        Q = X.transpose() @ X + lam * np.identity(d)
        c = injections[otherindices].transpose() @ projections[otherindices]  # - A'y
        betas = np.empty((Q.shape[0], c.shape[1]))
        for j in range(c.shape[1]):
            bjs, _ = scipy.optimize.nnls(Q, c[:, j])
            betas[:, j] = bjs  # bjs / norms

        # betas[np.where(np.isnan(betas))] = 0.
        # betas[np.where(betas == np.inf)] = 0.
        # betas[np.where(betas == np.nan)] = 0.

        # print(betas.shape)
        # print(injections[i:(i+1)].shape)

        pred = betas.transpose() @ injections[i:(i + 1)].transpose()  # homo_est.predict(injections[i:(i+1)])
        predictions[i] = np.squeeze(pred)

    return (predictions)


def get_loocv_predictions_nnlinear_nmf(projections, injections, n_components):
    # projections = reg_proj_vcount_norm
    # injections = reg_inj_vcount_norm
    projections = np.asarray(projections, dtype=np.float32)
    injections = np.asarray(injections, dtype=np.float32)
    nexp = projections.shape[0]
    predictions = projections.copy()

    homo_est = HomogeneousModel(kappa=np.inf)
    NMF = decomposition.NMF(n_components=n_components)

    recon_err = np.zeros(nexp)
    for i in range(nexp):
        # print(i)
        otherindices = np.setdiff1d(np.asarray(list(range(nexp))), i)
        NMF.fit(injections[otherindices])
        trans_inj = NMF.transform(injections[otherindices])
        homo_est.fit(trans_inj, projections[otherindices])

        test_inj = NMF.transform(injections[i:(i + 1)])
        tpred = homo_est.predict(test_inj)
        predictions[i] = tpred  # NMF.inverse_transform(tpred)
        recon_err[i] = NMF.reconstruction_err_

    return (predictions, recon_err)


def get_loocv_predictions_nnlinear_ic(projections, injections, rank, nfeatures):
    # this is not necessarily the best option but oh well
    number = rank

    projections = np.asarray(projections, dtype=np.float32)
    injections = np.asarray(injections, dtype=np.float32)
    nexp = projections.shape[0]
    conds = np.empty(nexp)
    predictions = projections.copy()
    homo_est = HomogeneousModel(kappa=np.inf)

    for i in range(nexp):
        print('exp', i)
        otherindices = np.setdiff1d(np.asarray(list(range(nexp))), i)
        injs = injections[otherindices]
        inds = svd_subset_selection(injs, number)
        # if len(inds) == 1:
        #    homo_est.fit(injs[:,[inds]], projections[otherindices])
        #    pred =  homo_est.predict(injections[i:(i+1)][:,[inds]])
        # else:
        homo_est.fit(injs[:, inds], projections[otherindices])
        pred = homo_est.predict(injections[i:(i + 1)][:, inds])
        predictions[i] = pred
        conds[i] = LA.cond(injs[:, inds])

    return (predictions, conds)


def get_loocv_predictions_nnlinear_number_inj(projections, injections, thresh, number):
    projections = np.asarray(projections, dtype=np.float32)
    injections = np.asarray(injections, dtype=np.float32)
    nexp = projections.shape[0]
    predictions = np.zeros(projections.shape)
    homo_est = HomogeneousModel(kappa=np.inf)

    for i in range(nexp):
        print('exp', i)
        otherindices = np.setdiff1d(np.asarray(list(range(nexp))), i)
        inj, inds = get_reduced_matrix_ninj(injections[otherindices], thresh, number)
        if inj.shape[1] > 0:
            homo_est.fit(inj, projections[otherindices])
            pred = homo_est.predict(injections[i:(i + 1)][:, inds])
            predictions[i] = pred

    return (predictions)



def get_loocv_predictions_nnlinear_number_inj_norm(projections, injections, thresh, number):
    projections = np.asarray(projections, dtype=np.float32)
    injections = np.asarray(injections, dtype=np.float32)
    nexp = projections.shape[0]
    predictions = np.zeros(projections.shape)
    homo_est = HomogeneousModel(kappa=np.inf)

    for i in range(nexp):
        print('exp', i)
        otherindices = np.setdiff1d(np.asarray(list(range(nexp))), i)
        inj, inds = get_reduced_matrix_ninj(injections[otherindices], thresh, number)
        if inj.shape[1] > 0:
            homo_est.fit(inj, projections[otherindices])
            pred = homo_est.predict(injections[i:(i + 1)][:, inds])
            predictions[i] = pred / np.linalg.norm(pred)

    return (predictions)



def get_loocv_predictions_nnlinear_pca(projections, injections, n_components):
    # projections = reg_proj_vcount_norm
    # injections = reg_inj_vcount_norm
    projections = np.asarray(projections, dtype=np.float32)
    injections = np.asarray(injections, dtype=np.float32)
    nexp = projections.shape[0]
    predictions = projections.copy()

    homo_est = HomogeneousModel(kappa=np.inf)
    SVD = decomposition.TruncatedSVD(n_components=n_components)

    exp_var = np.zeros(nexp)
    for i in range(nexp):
        print(i)
        otherindices = np.setdiff1d(np.asarray(list(range(nexp))), i)
        SVD.fit(injections[otherindices])
        trans_inj = SVD.transform(injections[otherindices])
        homo_est.fit(trans_inj, projections[otherindices])
        test_inj = SVD.transform(injections[i:(i + 1)])
        homo_est.fit(trans_inj, projections[otherindices])
        tpred = homo_est.predict(test_inj)
        predictions[i] = tpred
        # predictions[i] = SVD.inverse_transform(tpred.transpose())
        exp_var[i] = np.sum(SVD.explained_variance_)

    return (predictions, exp_var)


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

        output[sid] = np.zeros(np.append([len(np.unique(keys[:, i])) for i in range(keys.shape[1])], nexp))

        for j in range(keys.shape[0]):
            output[sid][tuple(keys[j])] = np.asarray(
                [get_loss_paper(true_dict[sid][tind[i]], prediction_dict[sid][tuple(keys[j])][pind[i]]) for i in
                 range(nexp)])

    return (output)


def get_best_hyperparameters(losses, keys):
    major_structure_ids = np.asarray(list(losses.keys()))
    nms = len(major_structure_ids)
    nkey = keys.shape[1]
    output = np.zeros((nms, nkey))
    for m in range(nms):
        print(m)
        sid = major_structure_ids[m]
        lvec = np.asarray([np.nanmean(losses[sid][key]) for key in keys])
        output[m] = keys[np.nanargmin(lvec)]
        # if len(np.where(np.isnan(np.nanmean(losses[sid][:,:], axis = 1)))[0]) < losses[sid].shape[0]:
        #    output[m] = np.nanargmin(np.nanmean(losses[sid][:,:], axis = 1))

    output = np.asarray(output, dtype=int)
    return (output)


def get_loss_best_hyp(losses, hyps):
    major_structure_ids = np.asarray(list(losses.keys()))
    nms = len(major_structure_ids)
    output = np.zeros(nms)
    for m in range(nms):
        sid = major_structure_ids[m]
        output[m] = np.nanmean(losses[sid][hyps[m], :])
    return (output)


# injection_vector = vdata[major_structure].injections[i]
