import numpy as np
from mcmodels.models.homogeneous import svd_subset_selection, HomogeneousModel
from mcmodels.regressors.nonparametric.nadaraya_watson import get_weights
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

# injection_vector = vdata[major_structure].injections[i]
