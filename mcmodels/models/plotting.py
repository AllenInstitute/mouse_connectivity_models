import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def plot_loss(meanloss, fontsize):
    ydim = meanloss.shape[0]
    ticks = np.arange(np.nanmin(meanloss), np.nanmax(meanloss))
    fig = plt.figure(figsize=(60, 40))
    ax1 = plt.subplot2grid((20, 20), (0, 0), colspan=19, rowspan=19)
    ax2 = plt.subplot2grid((20, 20), (19, 0), colspan=19, rowspan=1)
    ax3 = plt.subplot2grid((20, 20), (0, 19), colspan=1, rowspan=19)

    mask = np.zeros_like(meanloss)
    mask[np.tril_indices_from(mask)] = True

    sns.heatmap(
        meanloss,
        ax=ax1,
        annot=False,
        cmap="Greys",
        linecolor="b",
        cbar=False,
        annot_kws={"size": 20},
    )
    ax1.xaxis.tick_top()
    ax1.set_xticklabels(meanloss.columns, rotation=40, fontsize=40)
    ax1.set_yticklabels(meanloss.index, rotation=40, fontsize=40)
    sns.heatmap(
        (pd.DataFrame(meanloss.mean(axis=0))).transpose(),
        ax=ax2,
        annot=False,
        annot_kws={"size": 20},
        cmap="Greys",
        cbar=False,
        xticklabels=False,
        yticklabels=False,
    )
    sns.heatmap(
        pd.DataFrame(meanloss.mean(axis=1)),
        ax=ax3,
        cbar_kws={"fraction": 0.7, "pad": -0.1},
        annot=False,
        cmap="Greys",
        annot_kws={"size": 20},
        cbar=True,
        xticklabels=False,
        yticklabels=False,
    )
    # ax2hm.cbar.set
    cbar = ax3.collections[0].colorbar
    cbar.ax.tick_params(labelsize=70)
    ax1.set_xticklabels(ax1.get_xticklabels(), fontsize=fontsize, rotation=90)

    ax1.set_yticklabels(ax1.get_yticklabels(), fontsize=fontsize, rotation=0)
    return fig
