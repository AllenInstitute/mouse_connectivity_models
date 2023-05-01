"""
Module containing utility functions
"""
# Authors: Joseph Knox josephk@alleninstitute.org, Samson Koelle s@uberduck.ai
# License: Allen Institute Software License

import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib.pyplot import gcf
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
from sklearn.decomposition._nmf import NMF
import matplotlib.colors as colors
from einops import rearrange
from sklearn import metrics, cluster

# NOTE (Sam): this is a custom function that ignores short distances (see README on github)
from sklearn.decomposition import non_negative_factorization
from collections import Counter

from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache

mpl.rcParams["text.usetex"] = True
mpl.rcParams["text.latex.preamble"] = [r"\usepackage{amsmath}"]


def cv_nmf_missing(data, n_components, alpha=0.01, l1_ratio=1.0, p_holdout=0.3):
    missings = np.asarray(np.where(np.isnan(data))).transpose()
    presents = np.asarray(np.where(~np.isnan(data))).transpose()

    M = np.where(np.random.rand(presents.shape[0]) > p_holdout)[0]
    N = np.where(np.random.rand(presents.shape[0]) < p_holdout)[0]

    nmf = NMF(
        n_components=n_components,
        alpha=alpha,
        l1_ratio=l1_ratio,
        solver="mu",
        init="random",
        max_iter=500,
    )

    data_tr = data.copy()
    for i in range(len(M)):
        data_tr[presents[M[i], 0], presents[M[i], 1]] = np.nan

    data_te = data.copy()
    for i in range(len(N)):
        data_te[presents[N[i], 0], presents[N[i], 1]] = np.nan

    nmf.fit(data_tr)

    tr_nmf_embedding = nmf.transform(data_tr)
    te_nmf_embedding = nmf.transform(data_te)

    tr_nmf_recon = nmf.inverse_transform(tr_nmf_embedding)
    te_nmf_recon = nmf.inverse_transform(te_nmf_embedding)
    tr_err = np.nanmean((data_tr - tr_nmf_recon) ** 2)
    te_err = np.nanmean((data_te - te_nmf_recon) ** 2)

    return (tr_err, te_err)


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        "trunc({n},{a:.2f},{b:.2f})".format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)),
    )
    return new_cmap


def compute_nmf_replicates(wt_conn, distance_threshold, ncomp, nreps, dists):
    low_values = np.asarray(np.where(dists < distance_threshold)).transpose()

    data = np.asarray(fill_df_na(wt_conn, low_values))
    rss = np.nansum(data, axis=1)
    ids = np.where(rss > 0.0)[0]
    data = data[ids]

    components = np.zeros((nreps, ncomp, 1123))
    for r in range(nreps):
        print(r)
        nmf = NMF(
            n_components=ncomp,
            alpha=0.0002,
            l1_ratio=1.0,
            solver="mu",
            init="random",
            max_iter=500,
        )
        nmf.fit(data)
        components[r] = nmf.components_

    combined = rearrange(components, "r c p -> (r c) p")
    return combined, ids, data


def compute_rand_indices(combined):
    nclusts = np.asarray([10, 20, 30, 40, 50])
    nreps_cluster = 15

    results = np.zeros(nclusts.shape[0])
    for d in range(len(nclusts)):
        print(d)
        nclust = nclusts[d]
        cluster_reps = np.zeros((nreps_cluster, combined.shape[0]))
        for c in range(nreps_cluster):
            kmeans = cluster.KMeans(n_clusters=nclust)
            kmeans.fit(combined)
            clusts = kmeans.labels_[kmeans.labels_.argsort()]
            cluster_reps[c] = kmeans.labels_

            rands = np.zeros((nreps_cluster, nreps_cluster))
            rands[:] = np.nan
        for i in range(nreps_cluster):
            for j in range(nreps_cluster):
                if i != j:
                    rands[i, j] = metrics.adjusted_rand_score(
                        cluster_reps[i], cluster_reps[j]
                    )

        results[d] = np.nanmean(rands)

    rcomb = np.vstack([np.asarray(list(range(1, 6))) * 10, results])
    stab = pd.DataFrame(rcomb, index=["q", "Rand index"]).transpose()
    print(stab.transpose().to_latex())


def get_top_clusters(combined, nclust=30):
    kmeans = cluster.KMeans(n_clusters=nclust)
    kmeans.fit(combined)
    clusts = kmeans.labels_[kmeans.labels_.argsort()]
    cclusts = Counter(clusts)
    topclusts = np.asarray(list(cclusts.keys()))[
        np.asarray(list(cclusts.values())).argsort()
    ][-15:]
    top_clust_inds = np.where(np.isin(clusts, topclusts))[0]
    return top_clust_inds, kmeans, clusts, topclusts


def get_toplot(
    wt_conn,
    kmeans,
    ncomp,
    nreps,
    top_clust_inds,
    bottom_thresh,
    ool_ic_majors,
    targ_str_labels,
    clusts,
    combined,
):
    targs = np.asarray(list(wt_conn.columns))[:, 1]
    layers_targs = np.zeros(targs.shape[0], dtype=object)
    for i in range(targs.shape[0]):
        if np.asarray(list(targs))[i][-3:] == "2/3":
            layers_targs[i] = "2/3"
        if np.asarray(list(targs))[i][-1:] == "1":
            layers_targs[i] = "1"
        if np.asarray(list(targs))[i][-1:] == "4":
            layers_targs[i] = "4"
        if np.asarray(list(targs))[i][-1:] == "5":
            layers_targs[i] = "5"
        if np.asarray(list(targs))[i][-2:] == "6a":
            layers_targs[i] = "6a"
        if np.asarray(list(targs))[i][-2:] == "6b":
            layers_targs[i] = "6b"

    sel_labels = np.zeros(ncomp * nreps, dtype=str)
    sel_labels[top_clust_inds] = "Yes"
    sel_labels[np.setdiff1d(list(range(ncomp * nreps)), top_clust_inds)] = "No"
    rep_labels = np.asarray(
        np.repeat(list(range(10)), 15)[kmeans.labels_.argsort()], dtype=str
    )
    clust_labels = np.asarray(clusts, dtype=str)
    row_multi_ind = np.vstack([sel_labels, rep_labels, clust_labels])
    row_multi_ind = np.asarray(row_multi_ind, dtype=str)
    row_multi_ind_tuples = list(zip(*row_multi_ind))
    row_multi_ind_tuples = pd.MultiIndex.from_tuples(
        row_multi_ind_tuples, names=["Top 15 basis", "Replicate", "Cluster"]
    )

    sel_labels = row_multi_ind_tuples.get_level_values("Top 15 basis")
    sel_pal = sns.color_palette("Pastel1", n_colors=np.unique(sel_labels).shape[0])
    sel_lut = dict(zip(map(str, np.unique(sel_labels)), sel_pal))
    sel_colors = pd.Series(sel_labels, index=row_multi_ind_tuples).map(sel_lut)

    rep_labels = row_multi_ind_tuples.get_level_values("Replicate")
    rep_pal = sns.color_palette("tab20", n_colors=np.unique(rep_labels).shape[0])
    rep_lut = dict(zip(map(str, np.unique(rep_labels)), rep_pal))
    rep_colors = pd.Series(rep_labels, index=row_multi_ind_tuples).map(rep_lut)

    clust_labels = row_multi_ind_tuples.get_level_values("Cluster")
    clust_pal = sns.color_palette("Paired", np.unique(clust_labels).shape[0])
    clust_lut = dict(zip(map(str, np.unique(clust_labels)), clust_pal))
    clust_colors = pd.Series(clust_labels, index=row_multi_ind_tuples).map(clust_lut)

    arr = np.zeros((ncomp * nreps, 3), dtype=object)
    arr[:, 0] = sel_colors
    arr[:, 1] = rep_colors
    arr[:, 2] = clust_colors
    combined_colors = pd.DataFrame(arr, index=clust_colors.index)

    toplot = pd.DataFrame(
        np.log10(combined[kmeans.labels_.argsort()]), index=combined_colors.index
    )
    wt_conn.columns.names = np.asarray(["Hemisphere", "Structure"])
    hemi_labels = wt_conn.columns.get_level_values("Hemisphere")
    hemi_pal = sns.color_palette("cubehelix", n_colors=hemi_labels.unique().size)
    hemi_lut = dict(zip(map(str, hemi_labels.unique()), hemi_pal))
    hemi_colors = pd.Series(hemi_labels, index=wt_conn.columns).map(hemi_lut)

    targ_str_pal = sns.color_palette("tab20", n_colors=np.unique(targ_str_labels).size)
    targ_str_lut = dict(zip(map(str, np.unique(targ_str_labels)), targ_str_pal))
    targ_str_colors = pd.Series(targ_str_labels, index=wt_conn.columns).map(
        targ_str_lut
    )

    layer_labels_col = np.asarray(layers_targs, dtype=str)
    layer_pal = sns.color_palette("Set1", n_colors=6)
    layer_lut_col = dict(
        zip(map(str, np.unique(np.asarray(layer_labels_col, dtype=str))[1:]), layer_pal)
    )
    layer_colors = pd.Series(layer_labels_col, index=wt_conn.columns).map(layer_lut_col)

    component_labels = np.asarray(list(range(15)), dtype=str)
    component_pal = sns.color_palette("prism", n_colors=15)
    component_lut = dict(
        zip(map(str, np.unique(np.asarray(component_labels, dtype=str))), component_pal)
    )
    component_colors = pd.Series(component_labels, index=list(range(15))).map(
        component_lut
    )

    arr_targ = np.zeros((1123, 2), dtype=object)
    arr_targ[:, 0] = hemi_colors
    arr_targ[:, 1] = targ_str_colors
    combined_colors_targ = pd.DataFrame(arr_targ, index=wt_conn.columns)
    toplot.columns = combined_colors_targ.index
    # NOTE (Sam): should this be bottom_thresh or 0/-inf?
    toplot[toplot < bottom_thresh] = bottom_thresh

    return (
        toplot,
        combined_colors,
        combined_colors_targ,
        sel_labels,
        sel_lut,
        component_colors,
        targ_str_lut,
        layer_labels_col,
    )


def plot_nmf_replicates(
    toplot,
    combined_colors,
    combined_colors_targ,
    reo,
    color_str,
    hemis,
    color_hemi,
    sel_labels,
    major_structures,
    sel_lut,
):
    g = sns.clustermap(
        toplot,
        vmin=-6,
        vmax=0,
        row_cluster=False,
        col_cluster=False,
        row_colors=combined_colors,
        col_colors=combined_colors_targ,
        cbar_pos=[0.98, 0.1, 0.15, 0.2],
        dendrogram_ratio=0.001,
        cmap="Greys",
    )
    g.ax_col_colors.set_yticklabels(["Hemisphere", "Major Structure"], fontsize=20)
    g.ax_row_colors.set_xticklabels(
        ["Selected", "NMF replicate", "Cluster"], fontsize=20
    )
    g.ax_heatmap.set_xticks([])
    g.ax_heatmap.set_yticks([])
    g.ax_heatmap.set_yticklabels([])
    g.ax_heatmap.set_xticklabels([])
    g.ax_heatmap.set_xlabel("Targets", fontsize=30)
    g.ax_heatmap.set_ylabel("Components", fontsize=30)
    g.ax_cbar.set_xlabel("$\log( H)$", rotation=0, fontsize=20)

    ax = gcf().add_axes((0.55, 1.0, 0.0, 0.0))
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    for label in np.asarray(major_structures)[reo]:
        ax.bar(0, 0, color=color_str[label], label=label, linewidth=0.0)
    l4 = ax.legend(
        title="Major Structure",
        ncol=1,
        bbox_to_anchor=(1.2, 0.76),
        bbox_transform=gcf().transFigure,
        fontsize=15,
        title_fontsize=15,
    )

    ax2 = gcf().add_axes((0.55, 0.5, 0.0, 0.0))
    ax2.xaxis.set_visible(False)
    ax2.yaxis.set_visible(False)
    for label in hemis:
        ax2.bar(0, 0, color=color_hemi[label], label=label, linewidth=0.0)
    l4 = ax2.legend(
        title="Hemisphere",
        ncol=1,
        bbox_to_anchor=(1.2, 0.88),
        bbox_transform=gcf().transFigure,
        fontsize=15,
        title_fontsize=20,
    )

    ax2 = gcf().add_axes((0.56, 0.5, 0.0, 0.0))
    ax2.xaxis.set_visible(False)
    ax2.yaxis.set_visible(False)
    for label in sel_labels.unique():
        ax2.bar(0, 0, color=sel_lut[label], label=label, linewidth=0.0)
    l4 = ax2.legend(
        title="Selected",
        ncol=1,
        bbox_to_anchor=(1.2, 1.0),
        bbox_transform=gcf().transFigure,
        fontsize=15,
        title_fontsize=20,
    )

    for t in g.ax_cbar.get_yticklabels():
        t.set_fontsize(20)
    g.ax_col_colors.set_title("NMF archetype stability", fontsize=40)
    return g


def get_H_archetypes(
    ncomp, wt_conn, kmeans, topclusts, bottom_thresh, combined, toplot
):
    archetypes = np.zeros((ncomp, wt_conn.shape[1]))
    for c in range(ncomp):
        archetypes[c] = np.mean(
            combined[np.where(kmeans.labels_ == topclusts[c])], axis=0
        )

    log_thresh_archetypes = np.log10(archetypes)
    log_thresh_archetypes[np.where(toplot < bottom_thresh)] = bottom_thresh
    log_thresh_archetypes = pd.DataFrame(log_thresh_archetypes, columns=wt_conn.columns)
    return log_thresh_archetypes, archetypes


def get_W_archetypes(
    data, ncomp, archetypes, alpha, bottom_thresh, combined_colors_source, ids
):
    W, H, n_iter = non_negative_factorization(
        data,
        n_components=ncomp,
        init="custom",
        solver="mu",
        random_state=0,
        update_H=False,
        H=archetypes,
        l1_ratio=1,
        alpha=alpha,
        regularization="both",
    )
    W_pd = pd.DataFrame(W, index=combined_colors_source.index[ids])
    W_pd_log = np.log10(W_pd)
    W_pd_log[W_pd_log < bottom_thresh] = bottom_thresh
    return W_pd_log, W


def plot_H_archetypes(
    H_archetypes,
    combined_colors_targ,
    component_colors,
    targ_str_labels,
    targ_str_lut,
    color_hemi,
    hemis,
):
    g = sns.clustermap(
        H_archetypes,
        vmin=-6,
        figsize=(20, 8),
        vmax=0,
        col_cluster=False,
        row_cluster=False,
        col_colors=combined_colors_targ,
        cbar_pos=[1.03, 0.22, 0.05, 0.2],
        colors_ratio=0.06,
        dendrogram_ratio=0.001,
        # row_colors=component_colors,
        cmap="Greys",
    )

    g.ax_heatmap.set_xticks([])
    g.ax_heatmap.set_ylabel("Component " + r"$c$", fontsize=40)
    g.ax_heatmap.set_xlabel("Target " + r"$t$", fontsize=40)
    g.ax_heatmap.yaxis.tick_left()
    # g.ax_heatmap.set_yticks([])
    g.ax_col_colors.set_yticks([])
    g.ax_cbar.set_xlabel("$\log( H_{wt})$", rotation=0, fontsize=40)
    g.ax_cbar.set_position([1.22, 0.5, 0.1, 0.3])

    ax = gcf().add_axes((0.55, 1.0, 0.0, 0.0))
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    for label in np.unique(targ_str_labels):
        ax.bar(0, 0, color=targ_str_lut[label], label=label, linewidth=0.0)
    l4 = ax.legend(
        title="Major Structure",
        ncol=1,
        bbox_to_anchor=(1.2, 0.9),
        bbox_transform=gcf().transFigure,
        fontsize=15,
        title_fontsize=20,
    )

    ax2 = gcf().add_axes((0.55, 0.5, 0.0, 0.0))
    ax2.xaxis.set_visible(False)
    ax2.yaxis.set_visible(False)
    for label in hemis:
        ax2.bar(0, 0, color=color_hemi[label], label=label, linewidth=0.0)
    l4 = ax2.legend(
        title="Hemisphere",
        ncol=1,
        bbox_to_anchor=(1.1, 0.7),
        bbox_transform=gcf().transFigure,
        fontsize=15,
        title_fontsize=20,
    )

    for t in g.ax_cbar.get_yticklabels():
        t.set_fontsize(20)

    g.ax_col_colors.set_title("Projection archetypes", fontsize=40)

    return g


def get_layers(wt_conn, ids, source_str_colors):
    layers = np.zeros(wt_conn.shape[0], dtype=object)
    for i in range(wt_conn.shape[0]):
        if np.asarray(list(wt_conn.index))[i][-3:] == "2/3":
            layers[i] = "2/3"
        if np.asarray(list(wt_conn.index))[i][-1:] == "1":
            layers[i] = "1"
        if np.asarray(list(wt_conn.index))[i][-1:] == "4":
            layers[i] = "4"
        if np.asarray(list(wt_conn.index))[i][-1:] == "5":
            layers[i] = "5"
        if np.asarray(list(wt_conn.index))[i][-2:] == "6a":
            layers[i] = "6a"
        if np.asarray(list(wt_conn.index))[i][-2:] == "6b":
            layers[i] = "6b"

    layer_labels_row = layers[ids]  # col_multi_ind_tuples.get_level_values("Layer")
    layer_pal_row = sns.color_palette(
        "Set1", n_colors=6
    )  # sns.cubehelix_palette(layer_labels.unique().size, light=1., dark=0., reverse=False, start=0, rot=-1, hue = 1)
    layer_lut_row = dict(
        zip(
            map(str, np.unique(np.asarray(layer_labels_row, dtype=str))[1:]),
            layer_pal_row,
        )
    )
    layer_colors_row = pd.Series(layer_labels_row, index=wt_conn.index[ids]).map(
        layer_lut_row
    )
    arr_targ = np.zeros((len(ids), 2), dtype=object)
    arr_targ[:, 0] = source_str_colors[ids]
    arr_targ[:, 1] = layer_colors_row
    combined_colors_source2 = pd.DataFrame(arr_targ, index=wt_conn.index[ids])
    return layers, layer_colors_row, combined_colors_source2, layer_lut_row


def get_reconstructed_connectivities(
    W, H, data, wt_conn, dists, ool_ic_majors, ool_i_majors, ids
):
    dists = pd.DataFrame(dists)
    output = np.log10(W @ H)
    output[np.where(np.isnan(data))] = np.nan
    output = pd.DataFrame(output, columns=wt_conn.columns, index=wt_conn.index[ids])

    output.columns.names = np.asarray(["Hemisphere", "Structure"])
    hemi_labels = wt_conn.columns.get_level_values("Hemisphere")
    hemi_pal = sns.color_palette("cubehelix", n_colors=hemi_labels.unique().size)
    hemi_lut = dict(zip(map(str, hemi_labels.unique()), hemi_pal))
    hemi_colors = pd.Series(hemi_labels, index=wt_conn.columns).map(hemi_lut)

    targ_str_labels = ool_ic_majors  # wt_conn.columns.get_level_values("Structure")
    targ_str_pal = sns.color_palette("tab20", n_colors=np.unique(targ_str_labels).size)
    targ_str_lut = dict(zip(map(str, np.unique(targ_str_labels)), targ_str_pal))
    targ_str_colors = pd.Series(targ_str_labels, index=wt_conn.columns).map(
        targ_str_lut
    )

    source_str_labels = ool_i_majors
    source_str_pal = sns.color_palette(
        "tab20", n_colors=np.unique(source_str_labels).size
    )
    source_str_lut = dict(zip(map(str, np.unique(source_str_labels)), source_str_pal))
    source_str_colors = pd.Series(source_str_labels, index=wt_conn.index).map(
        source_str_lut
    )

    arr_targ = np.zeros((1123, 2), dtype=object)
    arr_targ[:, 0] = hemi_colors
    arr_targ[:, 1] = targ_str_colors
    combined_colors_targ = pd.DataFrame(arr_targ, index=wt_conn.columns)

    arr_source = np.zeros((564, 1), dtype=object)
    arr_source[:, 0] = source_str_colors
    combined_colors_source = pd.DataFrame(arr_source, index=wt_conn.index)

    combined_colors_targ.columns = np.asarray(["Hemisphere", "Major Structure"])
    combined_colors_source.columns = np.asarray(["Major Structure"])

    strs_unique = np.unique(ool_ic_majors)
    nstr = len(strs_unique)
    cs = sns.color_palette("tab20", nstr)

    cs_hemi = sns.color_palette("cubehelix", 2)

    color_hemi = {}
    hemis = np.asarray(["Ipsi", "Contra"])
    for i in range(2):
        # print(i)
        color_hemi[hemis[i]] = cs_hemi[i]

    low_values2 = np.asarray(np.where(dists.iloc[ids] < 15)).transpose()
    return (
        output,
        low_values2,
        combined_colors_targ,
        combined_colors_source,
        source_str_colors,
    )


def plot_reconstructed_connectivities(
    output,
    combined_colors_targ,
    combined_colors_source,
    new_cmap,
    reo,
    major_structures,
    color_str,
    low_values2,
    hemis,
    color_hemi,
):
    g = sns.clustermap(
        output,
        col_cluster=False,
        row_cluster=False,
        col_colors=combined_colors_targ,
        row_colors=combined_colors_source,
        figsize=(30, 20),
        cmap=new_cmap,
        vmin=-6,
        vmax=0.0,
        dendrogram_ratio=(0, 0.01),
        cbar_kws={"ticks": [-5, 0]},
        cbar_pos=[1.03, 0.16, 0.06, 0.2],
    )

    g.ax_heatmap.set_xticks([])
    g.ax_heatmap.set_yticks([])
    g.ax_heatmap.set_yticklabels([])
    g.ax_heatmap.set_xticklabels([])
    g.ax_col_colors.set_yticks([])
    g.ax_row_colors.set_xticks([])
    g.ax_heatmap.set_xticks([])
    g.ax_heatmap.set_yticks([])
    g.ax_heatmap.set_yticklabels([])
    g.ax_heatmap.set_xticklabels([])

    g.ax_heatmap.set_xlabel("Leaf " + r"$t \; (|\mathcal T| = 1123)$", fontsize=70)
    g.ax_heatmap.set_ylabel("Leaf " + r"$s \; (|\mathcal S| = 122)$", fontsize=70)

    ax = gcf().add_axes((0.55, 1.0, 0.0, 0.0))
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    for label in np.asarray(major_structures)[reo]:
        ax.bar(0, 0, color=color_str[label], label=label, linewidth=0.0)
    l4 = ax.legend(
        title="Major Structure",
        ncol=1,
        bbox_to_anchor=(1.11, 1.0),
        bbox_transform=gcf().transFigure,
        fontsize=30,
        title_fontsize=35,
    )

    ax2 = gcf().add_axes((0.55, 0.5, 0.0, 0.0))
    ax2.xaxis.set_visible(False)
    ax2.yaxis.set_visible(False)
    for label in hemis:
        ax2.bar(0, 0, color=color_hemi[label], label=label, linewidth=0.0)
    l4 = ax2.legend(
        title="Hemisphere",
        ncol=1,
        bbox_to_anchor=(1.11, 0.57),
        bbox_transform=gcf().transFigure,
        fontsize=30,
        title_fontsize=35,
    )
    g.ax_cbar.set_title(r"$\log (\widehat{ \mathcal C_{st}})$", rotation=0, fontsize=80)
    g.ax_cbar.set_yticklabels(g.ax_cbar.get_yticklabels(), fontsize=80)

    g.ax_col_colors.set_title(
        r"$\widehat{ \mathcal C}^N = WH \; ( q = 15)$", fontsize=70
    )
    for i in range(low_values2.shape[0]):
        g.ax_heatmap.add_patch(
            Rectangle(
                (low_values2[i][1], low_values2[i][0]),
                1,
                1,
                fill=False,
                edgecolor="red",
                lw=0.15,
            )
        )

    plt.margins(x=0, y=0)
    return g


def plot_W_archetypes(
    W_pd_log,
    combined_colors_source2,
    component_colors,
    major_structures,
    reo,
    color_str,
    layer_labels_col,
    layer_lut_row,
):
    g = sns.clustermap(
        W_pd_log,
        vmin=-6,
        figsize=(5, 13),
        vmax=0,
        col_cluster=False,
        row_cluster=False,
        cmap="Greys",
        row_colors=combined_colors_source2,
        col_colors=component_colors,
        cbar_pos=[1.1, 0.25, 0.15, 0.2],
        dendrogram_ratio=0.001,
    )
    g.ax_col_colors.set_title("Factorization into archetypes", fontsize=40)
    g.ax_heatmap.set_xticks([])
    g.ax_heatmap.set_xlabel("Component " + r"$c$", fontsize=40)
    g.ax_heatmap.set_ylabel("Source " + r"$s$", fontsize=40)
    g.ax_heatmap.set_yticks([])
    g.ax_row_colors.set_xticklabels(["Major Structure", "Layer"], fontsize=15)
    g.ax_cbar.set_title(r"$\log(W_{sc})$", rotation=0, fontsize=40)
    g.ax_col_colors.set_yticks([])

    ax = gcf().add_axes((0.55, 1.0, 0.0, 0.0))
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    for label in np.asarray(major_structures)[reo]:
        ax.bar(0, 0, color=color_str[label], label=label, linewidth=0.0)
    l4 = ax.legend(
        title="Major Structure",
        ncol=1,
        bbox_to_anchor=(1.3, 0.8),
        bbox_transform=gcf().transFigure,
        fontsize=15,
        title_fontsize=15,
    )

    ax = gcf().add_axes((0.55, 0.9, 0.0, 0.0))
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    for label in np.unique(layer_labels_col)[1:-1]:
        ax.bar(0, 0, color=layer_lut_row[label], label=label, linewidth=0.0)

    l4 = ax.legend(
        title="Layers",
        ncol=1,
        bbox_to_anchor=(1.2, 0.95),
        bbox_transform=gcf().transFigure,
        fontsize=15,
        title_fontsize=15,
    )

    for t in g.ax_cbar.get_yticklabels():
        t.set_fontsize(20)
    return g


def plot_distances(
    wt_conn,
    distance_threshold,
    dists,
    ool_ic_majors,
    ool_i_majors,
    major_structures,
    ia_map,
    reo,
):
    ool_ic_majors = np.asarray(
        [ia_map[ool_ic_majors[i]] for i in range(len(ool_ic_majors))]
    )
    ool_i_majors = np.asarray(
        [ia_map[ool_i_majors[i]] for i in range(len(ool_i_majors))]
    )

    low_values = np.asarray(np.where(dists < distance_threshold)).transpose()

    data = np.asarray(fill_df_na(wt_conn, low_values))
    rss = np.nansum(data, axis=1)
    ids = np.where(rss > 0.0)[0]
    data = data[ids]

    wt_conn.columns.names = np.asarray(["Hemisphere", "Structure"])
    hemi_labels = wt_conn.columns.get_level_values("Hemisphere")
    hemi_pal = sns.color_palette("cubehelix", n_colors=hemi_labels.unique().size)
    hemi_lut = dict(zip(map(str, hemi_labels.unique()), hemi_pal))
    hemi_colors = pd.Series(hemi_labels, index=wt_conn.columns).map(hemi_lut)

    targ_str_labels = ool_ic_majors  # wt_conn.columns.get_level_values("Structure")
    targ_str_pal = sns.color_palette("tab20", n_colors=np.unique(targ_str_labels).size)
    targ_str_lut = dict(zip(map(str, np.unique(targ_str_labels)), targ_str_pal))
    targ_str_colors = pd.Series(targ_str_labels, index=wt_conn.columns).map(
        targ_str_lut
    )

    source_str_labels = ool_i_majors
    source_str_pal = sns.color_palette(
        "tab20", n_colors=np.unique(source_str_labels).size
    )
    source_str_lut = dict(zip(map(str, np.unique(source_str_labels)), source_str_pal))
    source_str_colors = pd.Series(source_str_labels, index=wt_conn.index).map(
        source_str_lut
    )

    arr_targ = np.zeros((1123, 2), dtype=object)
    arr_targ[:, 0] = hemi_colors
    arr_targ[:, 1] = targ_str_colors
    combined_colors_targ = pd.DataFrame(arr_targ, index=wt_conn.columns)

    arr_source = np.zeros((564, 1), dtype=object)
    arr_source[:, 0] = source_str_colors
    combined_colors_source = pd.DataFrame(arr_source, index=wt_conn.index)

    dists = pd.DataFrame(dists, index=wt_conn.index, columns=wt_conn.columns)
    combined_colors_targ.columns = np.asarray(["Hemisphere", "Major Structure"])
    combined_colors_source.columns = np.asarray(["Major Structure"])

    strs_unique = np.unique(ool_ic_majors)
    nstr = len(strs_unique)
    cs = sns.color_palette("tab20", nstr)

    color_str = {}
    for i in range(nstr):
        # print(i)
        color_str[strs_unique[i]] = cs[i]

    cs_hemi = sns.color_palette("cubehelix", 2)
    color_hemi = {}
    hemis = np.asarray(["Ipsi", "Contra"])
    for i in range(2):
        color_hemi[hemis[i]] = cs_hemi[i]

    g = sns.clustermap(
        dists,
        col_cluster=False,
        row_cluster=False,
        figsize=(15, 15),
        cmap="Greys",
        col_colors=combined_colors_targ,
        row_colors=combined_colors_source,
        dendrogram_ratio=0.001,
    )
    g.ax_col_colors.set_yticks([])
    g.ax_row_colors.set_xticks([])
    g.ax_heatmap.set_xticks([])
    g.ax_heatmap.set_yticks([])
    g.ax_heatmap.set_yticklabels([])
    g.ax_heatmap.set_xticklabels([])
    g.ax_heatmap.set_xlabel("Targets (T = 1123)", fontsize=30)
    g.ax_heatmap.set_ylabel("Sources (S = 564)", fontsize=30)
    ax = gcf().add_axes((0.55, 1.0, 0.0, 0.0))
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    for label in np.asarray(major_structures)[reo]:
        ax.bar(0, 0, color=color_str[label], label=label, linewidth=0.0)
    l4 = ax.legend(
        title="Major Structure",
        ncol=1,
        bbox_to_anchor=(1.11, 0.8),
        bbox_transform=gcf().transFigure,
        fontsize=15,
        title_fontsize=20,
    )

    ax2 = gcf().add_axes((0.55, 0.5, 0.0, 0.0))
    ax2.xaxis.set_visible(False)
    ax2.yaxis.set_visible(False)
    for label in hemis:
        ax2.bar(0, 0, color=color_hemi[label], label=label, linewidth=0.0)
    l4 = ax2.legend(
        title="Hemisphere",
        ncol=1,
        bbox_to_anchor=(1.11, 0.5),
        bbox_transform=gcf().transFigure,
        fontsize=15,
        title_fontsize=20,
    )

    g.ax_cbar.set_xlabel("Distance " + r"$(100 \mu m)$", rotation=0, fontsize=20)
    g.ax_cbar.set_position([0.98, 0.3, 0.1, 0.1])
    g.ax_cbar.axhline(15, color="red", linewidth=5)
    g.ax_col_colors.set_title("Distances b/w structures", fontsize=40)
    for i in range(low_values.shape[0]):
        g.ax_heatmap.add_patch(
            Rectangle(
                (low_values[i][1], low_values[i][0]),
                1,
                1,
                fill=False,
                edgecolor="red",
                lw=0.15,
            )
        )

    return g, color_str, hemis, color_hemi, combined_colors_source, source_str_colors


def plot_cv_results(te_results, tr_results):
    splits = np.asarray(["Train", "Test"])
    colorsplits = {}
    colorsplits["Train"] = "red"
    colorsplits["Test"] = "blue"

    fig, ax = plt.subplots(figsize=(15, 15))
    sns.boxplot(data=np.log10(te_results[:, 1:]), ax=ax, color="blue")
    sns.boxplot(data=np.log10(tr_results[:, 1:]), ax=ax, color="red")
    # ax.set_xticklabels(list(range(1,11)))
    # ax.axvline(5, color="k", dashes=[2, 2])
    ax.text(8, -6, "Train " + r"$1_M = 1_{M(p)}$", color="red", fontsize=40)
    ax.text(8, -6.2, "Test " + r"$1_M = 1_{M(p)}^C$", color="blue", fontsize=40)
    ax.set_ylabel(
        r"$\log(\frac{1}{\|1_{M}\|_1 } \| 1_{M} \odot \widehat {\mathcal C_{wt,q}^N} - 1_{M} \odot \mathcal C_{wt}^N \|_2^2)$",
        fontsize=40,
    )
    ax.set_xlabel("q", fontsize=40)
    ax.set_xticklabels(np.asarray(list(range(1, 8))) * 10, fontsize=20)
    plt.yticks(fontsize=20)
    return fig


def cv_nmf_replicates(wt_conn, distance_threshold, leaf_distance_file):
    dists = np.load(leaf_distance_file)
    low_values = np.asarray(np.where(dists < distance_threshold)).transpose()

    data = np.asarray(fill_df_na(wt_conn, low_values))
    rss = np.nansum(data, axis=1)
    ids = np.where(rss > 0.0)[0]
    data = data[ids]

    train_err = []
    test_err = []
    nrep = 8
    tr_results = np.zeros((nrep, 8))
    te_results = np.zeros((nrep, 8))
    for n in tqdm(range(1, 9)):
        for r in range(nrep):
            tr_results[r, n - 1], te_results[r, n - 1] = cv_nmf_missing(
                data, n_components=n * 10, alpha=0.0002, l1_ratio=1.0, p_holdout=0.3
            )

    return tr_results, te_results


def fix_pdcsv(csv):
    csv_rownames = np.asarray(csv.iloc[:, 0])
    csv = csv.iloc[:, 1:]
    csv.index = csv_rownames
    return csv


def fill_df_na(data, indices):
    for i in range(indices.shape[0]):
        data.iloc[indices[i, 0], indices[i, 1]] = np.nan

    return data


def get_colors(structures, palettes, alpha):
    strs_unique = np.unique(structures)
    nstr = len(strs_unique)
    cs = sns.color_palette("Spectral", nstr)

    cs_alphas = np.hstack([np.asarray(cs), alpha * np.expand_dims(np.ones(12), 1)])
    color_str = {}
    for i in range(nstr):
        color_str[strs_unique[i]] = cs_alphas[i]

    output = np.zeros((len(structures), 4), dtype=float)
    for i in range(len(structures)):
        output[i] = np.asarray(color_str[structures[i]])

    return (output, color_str)


def cv_nmf_missing(data, n_components, alpha=0.01, l1_ratio=1.0, p_holdout=0.3):
    missings = np.asarray(np.where(np.isnan(data))).transpose()
    presents = np.asarray(np.where(~np.isnan(data))).transpose()

    M = np.where(np.random.rand(presents.shape[0]) > p_holdout)[0]
    N = np.where(np.random.rand(presents.shape[0]) < p_holdout)[0]

    nmf = NMF(
        n_components=n_components,
        alpha=alpha,
        l1_ratio=l1_ratio,
        solver="mu",
        init="random",
        max_iter=500,
    )

    data_tr = data.copy()
    for i in range(len(M)):
        data_tr[presents[M[i], 0], presents[M[i], 1]] = np.nan

    data_te = data.copy()
    for i in range(len(N)):
        data_te[presents[N[i], 0], presents[N[i], 1]] = np.nan

    nmf.fit(data_tr)

    tr_nmf_embedding = nmf.transform(data_tr)
    te_nmf_embedding = nmf.transform(data_te)

    tr_nmf_recon = nmf.inverse_transform(tr_nmf_embedding)
    te_nmf_recon = nmf.inverse_transform(te_nmf_embedding)
    tr_err = np.nanmean((data_tr - tr_nmf_recon) ** 2)
    te_err = np.nanmean((data_te - te_nmf_recon) ** 2)

    return (tr_err, te_err)


def fix_pdcsv(csv):
    csv_rownames = np.asarray(csv.iloc[:, 0])
    csv = csv.iloc[:, 1:]
    csv.index = csv_rownames
    return csv


def get_experiment_ids(mcc, structure_ids, cre=None):
    """Returns all experiment ids with injection in structure_ids.

    Parameters
    ----------
    mcc : MouseConnectivityCache object
    structure_ids: list
        Only return experiments that were injected in the structures provided here.
        If None, return all experiments.  Default None.
    cre: boolean or list
        If True, return only cre-positive experiments.  If False, return only
        cre-negative experiments.  If None, return all experiments. If list, return
        all experiments with cre line names in the supplied list. Default None.

    Returns
    -------
    List of experiment ids satisfying the parameters.
    """
    # filters injections by structure id or Decendent
    experiments = mcc.get_experiments(
        dataframe=False, cre=cre, injection_structure_ids=structure_ids
    )
    return [experiment["id"] for experiment in experiments]


def get_aligned_ids(st, list1, list2):
    output = np.empty(len(list1), dtype=int)
    for i in range(len(list1)):
        output[i] = np.intersect1d(st.ancestor_ids([list1[i]]), list2)[0]
    return output


def nonzero_unique(ar, **unique_kwargs):
    """np.unique returning only nonzero unique elements.

    Parameters
    ----------
    ar : array_like
        Input array. Unless `axis` is specified, this will be flattened if it
        is not already 1-D.

    return_index : bool, optional
        If True, also return the indices of `ar` (along the specified axis,
        if provided, or in the flattened array) that result in the unique array.

    return_counts : bool, optional
        If True, also return the number of times each unique item appears
        in `ar`.

    axis : int or None, optional
        The axis to operate on. If None, `ar` will be flattened beforehand.
        Otherwise, duplicate items will be removed along the provided axis,
        with all the other axes belonging to the each of the unique elements.
        Object arrays or structured arrays that contain objects are not
        supported if the `axis` kwarg is used.

    Returns
    -------
    unique : array
        Unique values sorted in the order in which they occur

    unique_indices : array, optional
        Indices of the first occurance of the unique values. Only returned if
        return_indices kwarg is specified as True.

    unique_counts : array
        Counts of the unique values. Only returned if return_counts kwarg is
        specified as True.

    See Also
    --------
    ordered_unique
    lex_ordered_unique
    """
    if "return_inverse" in unique_kwargs:
        raise NotImplementedError("returning inverse array not yet implemented")

    if np.all(ar):
        return np.unique(ar, **unique_kwargs)

    unique = np.unique(ar, **unique_kwargs)
    if unique_kwargs:
        return map(lambda x: x[1:], unique)

    return unique[1:]


def ordered_unique(ar, **unique_kwargs):
    """np.unique in the order in which the unique values occur.

    Similar outuput to pd.unique(), although probably not as fast.

    Parameters
    ----------
    ar : array_like
        Input array. Unless `axis` is specified, this will be flattened if it
        is not already 1-D.

    return_index : bool, optional
        If True, also return the indices of `ar` (along the specified axis,
        if provided, or in the flattened array) that result in the unique array.

    return_counts : bool, optional
        If True, also return the number of times each unique item appears
        in `ar`.

    axis : int or None, optional
        The axis to operate on. If None, `ar` will be flattened beforehand.
        Otherwise, duplicate items will be removed along the provided axis,
        with all the other axes belonging to the each of the unique elements.
        Object arrays or structured arrays that contain objects are not
        supported if the `axis` kwarg is used.

    Returns
    -------
    unique : array
        Unique values sorted in the order in which they occur

    unique_indices : array, optional
        Indices of the first occurrence of the unique values. Only returned if
        return_indices kwarg is specified as True.

    unique_counts : array
        Counts of the unique values. Only returned if return_counts kwarg is
        specified as True.

    See Also
    --------
    nonzero_unique
    lex_ordered_unique
    """
    if "return_inverse" in unique_kwargs:
        raise NotImplementedError("returning inverse array not yet implemented")

    _return_index = unique_kwargs.pop("return_index", False)
    unique = np.unique(ar, return_index=True, **unique_kwargs)

    # need indices (always @ index 1)
    unique = list(unique)
    indices = unique[1] if _return_index else unique.pop(1)
    permutation = np.argsort(indices)

    if unique_kwargs or _return_index:
        return map(lambda x: x[permutation], unique)

    return unique[0][permutation]


def lex_ordered_unique(ar, lex_order, allow_extra=False, **unique_kwargs):
    """np.unique in a given order.

    Parameters
    ----------
    ar : array_like
        Input array. Unless `axis` is specified, this will be flattened if it
        is not already 1-D.

    return_index : bool, optional
        If True, also return the indices of `ar` (along the specified axis,
        if provided, or in the flattened array) that result in the unique array.

    return_counts : bool, optional
        If True, also return the number of times each unique item appears
        in `ar`.

    axis : int or None, optional
        The axis to operate on. If None, `ar` will be flattened beforehand.
        Otherwise, duplicate items will be removed along the provided axis,
        with all the other axes belonging to the each of the unique elements.
        Object arrays or structured arrays that contain objects are not
        supported if the `axis` kwarg is used.

    Returns
    -------
    unique : array
        Unique values sorted in the order in which they occur

    unique_indices : array, optional
        Indices of the first occurrence of the unique values. Only returned if
        return_indices kwarg is specified as True.

    unique_counts : array
        Counts of the unique values. Only returned if return_counts kwarg is
        specified as True.

    See Also
    --------
    nonzero_unique
    ordered_unique
    """
    if "return_inverse" in unique_kwargs:
        raise NotImplementedError("returning inverse array not yet implemented")

    if len(set(lex_order)) < len(lex_order):
        raise ValueError("lex_order must not contain duplicates")

    unique = np.unique(ar, **unique_kwargs)
    if not unique_kwargs:
        unique = (unique,)

    if len(unique[0]) < len(lex_order):
        if allow_extra:
            # view, does not write to array lex_order
            # cast to np.array in order to index with boolean array
            lex_order = np.array(lex_order)[np.isin(lex_order, unique[0])]
        else:
            raise ValueError(
                "lex_order contains elements not found in ar, "
                "call with allow_extra=True"
            )

    # generate a permutation order for unique
    permutation = np.argsort(np.argsort(lex_order))

    if len(unique) > 1:
        return tuple(map(lambda x: x[permutation], unique))

    return unique[0][permutation]


def padded_diagonal_fill(arrays):
    """Returns array filled with uneven arrays padding with zeros.

    Arrays are placed in the return array such that each row/column only
    contains the elements of a single array. Can be thought of as representing
    disconnected subgraphs.

    Parameters
    ----------
    arrays : list
        List of 2D arrays with which to fill the return array.

    Returns
    -------
    padded : array
        Return array containing each of the input arrays, padded with zeros.
    """
    shapes = [x.shape for x in arrays]
    padded = np.zeros(tuple(map(sum, zip(*shapes))))

    i, j = 0, 0
    for (n_rows, n_cols), arr in zip(shapes, arrays):
        # fill padded with arr
        padded[i : i + n_rows, j : j + n_cols] = arr

        i += n_rows
        j += n_cols

    return padded


def squared_norm(arr):
    """Compute the square frobenius/vector norm.

    Parameters
    ----------
    arr : np.ndarray
        Array of which we compute the norm.

    Returns
    -------
    norm: float
    """
    arr = arr.ravel(order="K")
    return np.dot(arr, arr)


def unionize(volume, key, return_regions=False):
    """Unionize voxel data to regional data.

    Parameters
    ----------
    volume : array, shape (n, m)
        Possibly stacked flattened volume(s) such as projection densities or
        injection densities.
    key : array, shape (m,)
        1D Array with length equal to the number of columns in volume. This array
        has integer values corresponding to the region to which each voxel belongs.
    return_regions : boolean, optional (default: False)
        If True, return an array of the unique values of key in addition to the
        unionized volume array.

    Returns
    -------
    result : array, shape (n, len(unique(key)))
        The unionized volume.
    regions : array, optional, shape (len(unique(key)),)
        The unique values of key.
    """
    volume = np.atleast_2d(volume)
    if volume.shape[1] != key.size:
        raise ValueError(
            "volume (%s) and key (%s) shapes are incompatible"
            % (volume.shape[1], key.size)
        )

    regions = nonzero_unique(key)
    result = np.empty((volume.shape[0], regions.size))
    for j, k in enumerate(regions):
        result[:, j] = volume[:, np.where(key == k)[0]].sum(axis=1)

    if return_regions:
        return result, regions

    return result
