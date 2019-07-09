from __future__ import division
import os
import logging
import operator as op

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as ticker
import matplotlib.colorbar as colorbar
import matplotlib.gridspec as gridspec
import allensdk.core.json_utilities as ju

from mcmodels.core import VoxelModelCache

from helpers import get_structure_map, plot_pt_line, plot_heatmap

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
TOP_DIR = os.path.join(FILE_DIR, '..', '..')

INPUT_JSON = os.path.join(TOP_DIR, 'input.json')
OUTPUT_FILE = os.path.join(FILE_DIR, 'regional_matrix.png')

STRUCTURE_LABELS = ["Iso-\ncortex", "OLF", "HPF", "CTXsp", "STR", "PAL",
                    "Thal", "Hypo-\nthal", "Mid-\nbrain", "Pons", "Medulla", "CB"]

FIGSIZE = (10, 5.5)
SAVEFIG_KWARGS = dict(dpi=200, transparent=True, bbox_inches='tight')
CBAR_KWS = dict(ticks=[1e-5, 1e-4, 1e-3], orientation='horizontal', fraction=0.5,
                format=ticker.LogFormatterExponent())
GRID_KWS = dict(nrows=3,
                ncols=3,
                wspace=0.0,
                hspace=0.0,
                height_ratios=(0.03, 0.10, 0.87),
                width_ratios=(0.06, 0.01, 0.93))
HEATMAP_KWS = dict(vmin=1e-5,
                   vmax=10**-2.5,
                   cmap="CMRmap",
                   norm="LogNorm",
                   epsilon=1e-10,
                   alpha=0.85,
                   cbar=True,
                   xticklabels=0,
                   yticklabels=0)


def get_hex_map(mcc):
    tree = mcc.get_structure_tree()
    to_hex = lambda x: colors.to_hex(tuple(map(lambda y: y/255, x)))

    return tree.value_map(lambda x: x["acronym"], lambda y: to_hex(y["rgb_triplet"]))


def get_hex_vals(mcc, structures):
    hex_map = get_hex_map(mcc)
    hex_vals = map(hex_map.get, structures)

    return list(hex_vals)


def get_cmap(hex_vals, rep=1):
    return colors.ListedColormap(rep*hex_vals)


def get_structure_bounds(df, structure_map):
    #TODO: breakup for each
    rows = pd.Series(df.index.map(structure_map.get))
    cols = pd.Series(tuple(zip(df.columns.get_level_values(0),
                               df.columns.get_level_values(1).map(structure_map.get))))

    def get_majors(series):
        counts = series.value_counts()
        return [0] + list(counts[series.unique()].cumsum())

    return get_majors(rows), get_majors(cols)


def fill_index(row_ax, col_ax, df, structures, labels, mcc):
    #TODO: breakup for each
    structure_map = get_structure_map(mcc, structures)

    hex_vals = get_hex_vals(mcc, structures)
    row_cmap = get_cmap(hex_vals)
    col_cmap = get_cmap(hex_vals, rep=2)

    row_bounds, col_bounds = get_structure_bounds(df, structure_map)

    # fill
    row_norm = colors.BoundaryNorm(row_bounds, row_cmap.N)
    row_cbar = colorbar.ColorbarBase(row_ax, cmap=row_cmap, norm=row_norm, boundaries=row_bounds,
                                     orientation='vertical', spacing='proportional',
                                     ticks=[])

    col_norm = colors.BoundaryNorm(col_bounds, col_cmap.N)
    col_cbar = colorbar.ColorbarBase(col_ax, cmap=col_cmap, norm=col_norm, boundaries=col_bounds,
                                     orientation='horizontal', spacing='proportional',
                                     ticks=[])

    #TODO: breakup
    # midpoints
    def get_minors(l):
        return list(map(lambda x: x/2, map(op.add, l[1:], l[:-1])))

    # annotate
    row_minors = get_minors(row_bounds)
    col_minors = get_minors(col_bounds)


    kws = dict(fontsize=8, ha='center', va='center')
    rotate_labels = ["OLF", "HPF", "CTXsp", "STR", "PAL", "Pons"]

    for y, label in zip(row_minors, labels):
        row_ax.text(0.5, y/max(row_bounds), label, **kws)

    for x, label in zip(col_minors, 2*labels):
        rotation = 90 if label in rotate_labels else 0
        col_ax.text(x/max(col_bounds), 0.5, label, rotation=rotation, **kws)


def remove_ticks(ax):
    # turn off tick labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])


def add_axis_labels(source_ax, target_ax):
    source_ax.text(-0.3, 0.5, "Source Region", rotation=90, va='center', ha='center',
                   fontsize=12)

    # targets
    target_ax.text(0.15, 1.2, "Ipsilateral Target", fontsize=12)
    target_ax.text(0.65, 1.2, "Contralateral Target", fontsize=12)



def plot(df, structures, labels, mcc, grid_kws, cbar_kws, heatmap_kws, figsize=None):

    # set up plotting en
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(**grid_kws)


    cbar_ax = fig.add_subplot(gs[0,0])
    source_ax = fig.add_subplot(gs[2, 0:2])
    target_ax = fig.add_subplot(gs[0:2, 2])
    heatmap_ax = fig.add_subplot(gs[2, 2])


    # plot heatmap
    plot_heatmap(df, heatmap_ax, cbar_ax, cbar_kws=cbar_kws, **heatmap_kws)
    remove_ticks(heatmap_ax)

    # plot line on heatmap
    plot_pt_line(df, heatmap_ax)

    # fill
    fill_index(source_ax, target_ax, df, structures, labels, mcc)
    source_ax.invert_yaxis()

    # annotate
    add_axis_labels(source_ax, target_ax)

    # cbar title
    cbar_ax.set_xlabel("Log$_{10}$", labelpad=0.1)

    # save
    return fig


def main():
    input_data = ju.read(INPUT_JSON)

    structures = input_data.get('structures')
    manifest_file = input_data.get('manifest_file')
    manifest_file = os.path.join(TOP_DIR, manifest_file)

    log_level = input_data.get('log_level', logging.DEBUG)
    logging.getLogger().setLevel(log_level)

    # get cache, metric
    logging.debug("loading regional matrix")
    cache = VoxelModelCache(manifest_file=manifest_file)
    df_metric = cache.get_normalized_connection_density(dataframe=True)

    # plot
    fig = plot(df_metric, structures, STRUCTURE_LABELS, cache, GRID_KWS, CBAR_KWS,
               HEATMAP_KWS, figsize=FIGSIZE)

    fig.savefig(OUTPUT_FILE, **SAVEFIG_KWARGS)
    plt.close(fig)


if __name__ == "__main__":
    main()
