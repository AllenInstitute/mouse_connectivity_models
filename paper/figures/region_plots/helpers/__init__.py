import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as colors


def get_structure_map(cache, structures):
    """Get a dictionary mapping from regions to structure ids (ancestors)"""
    tree = cache.get_structure_tree()
    # NOTE
    aid_map = tree.get_id_acronym_map()
    ida_map = tree.value_map(lambda x: x['id'], lambda y: y['acronym'])

    structure_ids = [aid_map[x] for x in structures]
    d = dict(zip(structure_ids, tree.descendant_ids(structure_ids)))

    return {ida_map[x] : ida_map[k] for k, v in d.items() for x in v}


def plot_pt_line(df, ax):
    n = df.shape[0]
    ax.plot(2*[n], [0, n], 'w')


def plot_heatmap(df, ax, cbar_ax, cbar_kws=None, norm=None, cmap=None,
                 vmin=None, vmax=None, epsilon=0.0, **kwargs):
    # get
    if cmap is not None:
        cmap = getattr(plt.cm, cmap)
    if norm is not None:
        norm = getattr(colors, norm)(vmin=vmin, vmax=vmax)

    # for log(x + e)
    df += epsilon

    # plot
    ax = sns.heatmap(df.values, ax=ax, cbar_ax=cbar_ax, cmap=cmap, norm=norm,
                     vmin=vmin, vmax=vmax, cbar_kws=cbar_kws, **kwargs)
