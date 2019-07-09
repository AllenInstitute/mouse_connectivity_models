import os
import logging
import argparse

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import allensdk.core.json_utilities as ju

from mcmodels.core import VoxelModelCache

from helpers import get_structure_map, plot_pt_line, plot_heatmap

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
TOP_DIR = os.path.join(FILE_DIR, '..', '..')

INPUT_JSON = os.path.join(TOP_DIR, 'input.json')
OUTPUT_FILE = os.path.join(FILE_DIR, 'cortical_matrix.png')

CORTEX_STRUCTURE = 'Isocortex'
STRUCTURES = tuple([CORTEX_STRUCTURE])

FIGSIZE = (10, 5)
SAVEFIG_KWARGS = dict(dpi=300, transparent=True, bbox_inches='tight')
CBAR_KWS = dict(ticks=[1e-5, 1e-4, 1e-3])
GRID_KWS = dict(nrows=1,
                ncols=2,
                width_ratios=(0.98, 0.02),
                wspace=0.01)
HEATMAP_KWS = dict(vmin=1e-5,
                   vmax=10**-2.5,
                   cmap="CMRmap",
                   norm="LogNorm",
                   epsilon=1e-10,
                   alpha=0.85,
                   cbar=True)


def subset_df(df, structures, mcc):
    """ subsets to structures """
    structure_map = get_structure_map(mcc, structures)
    valid_structures = frozenset(structure_map.keys())

    sources = [x for x in df.index.values if x in valid_structures]
    targets = [x for x in df.columns.values if x[1] in valid_structures]

    return df.loc[sources, targets]

def update_heatmap_kws(df, heatmap_kws):
    xticklabels = df.columns.get_level_values(1)
    yticklabels = df.index

    heatmap_kws.update(dict(xticklabels=xticklabels, yticklabels=yticklabels))
    return heatmap_kws


def update_tickparams(ax):
    ax.xaxis.set_tick_params(labelrotation=315, top=False, bottom=True,
                             labeltop=False, labelbottom=True)


def add_axis_labels(ax):
    ax.text(-0.1, 0.5, "Source Region", rotation=90, va='center', ha='center',
            fontsize=12, transform=ax.transAxes)

    # targets
    ax.text(0.14, 1.05, "Ipsilateral Target", va='center', fontsize=12, transform=ax.transAxes)
    ax.text(0.62, 1.05, "Contralateral Target",va='center',  fontsize=12, transform=ax.transAxes)


def get_savename(plot_dir, structures, metric):
    return os.path.join(plot_dir, "%s_%s.png" % ('-'.join(structures), metric))


def plot(df, structures, mcc, grid_kws, cbar_kws, heatmap_kws, figsize=None):

    # set up plotting en
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(**grid_kws)

    # axes
    heatmap_ax = fig.add_subplot(gs[0])
    cbar_ax = fig.add_subplot(gs[1])

    # subset to structures
    df = subset_df(df, structures, mcc)

    # get labels
    heatmap_kws = update_heatmap_kws(df, heatmap_kws)

    # plot heatmap
    plot_heatmap(df, heatmap_ax, cbar_ax, cbar_kws=cbar_kws, **heatmap_kws)


    update_tickparams(heatmap_ax)

    # plot line on heatmap
    plot_pt_line(df, heatmap_ax)


    # NOTE: xticks are the ones we turn off
    xticks = heatmap_ax.xaxis.get_major_ticks()
    if len(xticks) % 2:
        xticks = xticks[::2]
    else:
        # odd
        mid = len(xticks)//2
        xticks = xticks[:mid:2] + xticks[mid::2]

    for tick in xticks:
        tick.set_visible(False)

    for tick in heatmap_ax.yaxis.get_major_ticks()[1::2]:
        tick.set_visible(False)

    heatmap_ax.set_xticklabels(heatmap_ax.get_xticklabels(), ha='left')

    # annotate
    add_axis_labels(heatmap_ax)

    return fig


def main():
    input_data = ju.read(INPUT_JSON)

    manifest_file = input_data.get('manifest_file')
    manifest_file = os.path.join(TOP_DIR, manifest_file)

    log_level = input_data.get('log_level', logging.DEBUG)
    logging.getLogger().setLevel(log_level)

    # get cache, metric
    logging.debug("loading regional matrix")
    cache = VoxelModelCache(manifest_file=manifest_file)
    df_metric = cache.get_normalized_connection_density(dataframe=True)

    # plot
    fig = plot(df_metric, STRUCTURES, cache, GRID_KWS, CBAR_KWS,
               HEATMAP_KWS, figsize=FIGSIZE)

    fig.savefig(OUTPUT_FILE, **SAVEFIG_KWARGS)
    plt.close(fig)


if __name__ == "__main__":
    main()
