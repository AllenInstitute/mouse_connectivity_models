from __future__ import division
import os
import logging
import itertools

import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as patheffects
import allensdk.core.json_utilities as ju

from mcmodels.core import VoxelModelCache

from helpers.utils import get_cortical_df, get_pt, get_distances, to_dataframe
#from helpers.fit_distance_dependance import
from helpers.loglog_regplot import loglog_regplot

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
TOP_DIR = os.path.join(FILE_DIR, '..', '..')

INPUT_JSON = os.path.join(TOP_DIR, 'input.json')
OUTPUT_DIR = FILE_DIR


def _get_plotting_df(full, cortex):
    def _dataframe_it(x, division=""):
        d, w = x
        data = {"Distance (mm)":d, "Weight":w}
        return pd.DataFrame(data=data)

    df_full = _dataframe_it(full, division="Whole-Brain")
    df_cortex = _dataframe_it(cortex, division="Cortical-Cortico")

    return df_full, df_cortex


def rc_axes_off():
    x_pos = ("top", "bottom")
    y_pos = ("left", "right")

    spines = ["axes.spines." + x for x in itertools.chain(x_pos, y_pos)]
    xticks = ["xtick." + x for x in x_pos]
    yticks = ["ytick." + x for x in y_pos]

    params = (spines, xticks, yticks)
    return {param : False for param in itertools.chain(*params)}


def plot_log_fit(a, fit, ax=None, vertical=False, color="#282828"):
    def pdf(x):
        return fit.pdf(x, *params)

    params = fit.fit(a[a > 0], floc=0)

    clip = ax.get_ylim() if vertical else ax.get_xlim()
    x = np.logspace(np.log10(clip[0]), np.log10(clip[1]), 100)
    y = pdf(x)

    fit = stats.norm
    params = fit.fit(np.log10(a[a>0]))
    y = fit.pdf(np.log10(x), *params)

    y /= y.max()
    y *= ax.get_xlim()[1] if vertical else ax.get_ylim()[1]

    if vertical:
        x, y = y, x

    label = '$(\mu, \sigma)$ : (%.2f, %.2f)' % params
    fit = ax.plot(x, y, color=color, label=label)
    return fit


def plot(data, colors, x="x", y="y", reg_kws=None, dist_kws=None):
    """tuples"""
    # times font
    sns.set_style("white", {'font.family': [u'serif'], 'font.serif': [u'Times']})
    def plot_regression(x=None, y=None, data=None, ax=None, label="", **kwargs):
    # plot multiple lin regression
        ax, betas = loglog_regplot(x=x, y=y, data=data, ax=ax, **kwargs)
        ax.lines[-1].set_label(r"%s $(\beta_1 = %.2f)$"
                               % (label, betas[1]))
        return ax

    def get_bins(x, n=100):
        x_range = x.min(), x.max()
        x_range = list(map(np.log, x_range))
        return np.logspace(x_range[0], x_range[1], n)

    def plot_dist(x, bins=None, ax=None, fit=None, fit_kws=dict(), vertical=False, x_max=1, **kwargs):
        hist = sns.distplot(x, bins=bins, ax=ax, vertical=vertical, **kwargs)
        if fit is not None:
            color = kwargs.pop('color', '#282828')
            fit = plot_log_fit(x, fit, ax=ax, vertical=vertical, color=color)

        return fit

    def get_dist_tick_params():
        ticks = ('bottom', 'top', 'left', 'right')
        labels = ['label' + x for x in ticks]
        params = {x : False for x in itertools.chain(ticks, labels)}
        params.update(dict(axis='both', which='both'))
        return params

    fig = plt.figure(figsize=(8,3))
    gs = gridspec.GridSpec(1, 2, width_ratios=(0.8, 0.2), wspace=0.0)
    reg_ax = fig.add_subplot(gs[0, 0])
    reg_ax.set(xscale='log', yscale='log', ylim=(1e-13, 1))
    dist_ax = fig.add_subplot(gs[0, 1], sharey=reg_ax)

    # HACK
    labels = "Whole-Brain", "Isocortex"

    # plot regression
    lines = []
    for frame, color, label in zip(data, colors, labels):
        reg_kws.update(dict(color=color))
        ax = plot_regression(x, y, frame, reg_ax, label=label, **reg_kws)
        lines.append(ax.lines[-1])

    reg_ax.legend(lines[::-1], [l.get_label() for l in lines[::-1]])

    # plot hists
    clip = reg_ax.get_ylim()
    bins = np.logspace(np.log10(clip[0]), np.log10(clip[1]), 40)
    tick_params = get_dist_tick_params()

    fits = []
    dist_kws.update(dict(fit=stats.lognorm))
    for i, (frame, color) in enumerate(zip(data, colors)):
        # duplicate axes
        ax = dist_ax.twiny() if i != 0 else dist_ax
        ax.set(ylim=clip)
        sns.despine(ax=ax, bottom=True)
        ax.tick_params(**tick_params)

        dist_kws.update(dict(color=color))
        fit = plot_dist(frame[y], ax=ax, bins=bins, **dist_kws)
        fits.append(fit[0])

    plt.legend(fits[::-1], [f.get_label() for f in fits[::-1]])

    return fig


if __name__ == "__main__":

    input_data = ju.read(INPUT_JSON)

    manifest_file = input_data.get('manifest_file')
    manifest_file = os.path.join(TOP_DIR, manifest_file)

    log_level = input_data.get('log_level', logging.DEBUG)
    logging.getLogger().setLevel(log_level)

    # configure
    colors = sns.color_palette(n_colors=2)

    # get cache, metric
    cache = VoxelModelCache(manifest_file=manifest_file)
    df_metric = cache.get_normalized_connection_density(dataframe=True)

    logging.debug("getting cortical network")
    df_cortex = get_cortical_df(df_metric, cache)

    # region acs
    region_acs = df_metric.index.values

    logging.debug("computing distances")
    d = get_distances(region_acs, cache)
    d = to_dataframe(d, df_metric.index, df_metric.columns)
    d_cortex = get_cortical_df(d, cache)

    # get projection types
    full_ipsi = get_pt((d, df_metric), thresh=0)
    full_contra = get_pt((d, df_metric), thresh=0, pt="contra")
    cortex_ipsi = get_pt((d_cortex, df_cortex), thresh=0)
    cortex_contra = get_pt((d_cortex, df_cortex), thresh=0, pt="contra")


    # get plotting d
    ipsi = _get_plotting_df(full_ipsi, cortex_ipsi)
    contra = _get_plotting_df(full_contra, cortex_contra)

    line_kws = dict(path_effects=[patheffects.Normal(),
                                  patheffects.SimpleLineShadow()])
    scatter_kws = dict(alpha=0.05)
    reg_kws = dict(truncate=False, ci=None, line_kws=line_kws, scatter_kws=scatter_kws)
    dist_kws = dict(vertical=True, kde=False, axlabel=False)

    logging.debug("plotting densities")
    names = "ipsi", "contra"
    titles = "Ipsilateral", "Contralateral"
    for data, name, title in zip((ipsi, contra), names, titles):
        fig = plot(data, colors, *data[0].columns, reg_kws=reg_kws, dist_kws=dist_kws)
        fig.suptitle(title, x=0.03, ha="left")

        filename = os.path.join(OUTPUT_DIR, "distance_dependence_%s.png" % name)
        fig.savefig(filename, dpi=200, transparent=False, bbox_inches='tight')
        plt.close(fig)
