'''
Script for generating the GIF in the README

See the documentation for more examples and API descriptions:
http://mouse-connectivity-models.readthedocs.io/en/latest/
'''
import os
import logging
import subprocess

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm

from mcmodels.core import Mask, VoxelModelCache
from mcmodels.core.cortical_map import CorticalMap

logger = logging.getLogger(name=__name__)

# file path where the data files will be downloaded
MANIFEST_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'connectivity', 'mcmodels_manifest.json')
OUTPUT_DIR = 'images'
GIF_CONVERT_COMMAND = 'convert -delay 3x100 -size 50x50 *.png output.gif'


def main():
    # caching object for downloading/loading connectivity/model data
    cache = VoxelModelCache(manifest_file=MANIFEST_FILE)

    # load voxel model
    logging.info('loading voxel array')
    voxel_array, source_mask, target_mask = cache.get_voxel_connectivity_array()
    reference_shape = source_mask.reference_space.annotation.shape
    vmax = 1.2 * np.percentile(voxel_array.nodes, 99)

    # 2D Cortical Surface Mapper
    mapper = CorticalMap(projection='top_view')

    # colormaps
    cmap_view = matplotlib.cm.viridis
    cmap_pixel = matplotlib.cm.spring
    cmap_view.set_bad(alpha=0)
    cmap_pixel.set_bad(alpha=0)

    # only want R hemisphere
    lookup = mapper.view_lookup.copy().T # transpose for vertical pixel query
    lookup[:lookup.shape[0]//2, :] = -1

    # dict(2D lookup_value -> avg(path))
    logging.info('beginning image creation')
    for i, val in enumerate(lookup[lookup > -1]):
        # get the mean path voxel
        path = mapper.paths[val][mapper.paths[val].nonzero()]
        path = np.vstack([np.unravel_index(x, reference_shape) for x in path])
        voxel = tuple(map(int, path.mean(axis=0)))

        try:
            row_idx = source_mask.get_flattened_voxel_index(voxel)
        except ValueError:
            logging.warning('voxel %s not in mask', voxel)
        else:
            # get voxel expression
            volume = target_mask.fill_volume_where_masked(np.zeros(reference_shape),
                                                          voxel_array[row_idx])

            # map to cortical surface
            flat_view = mapper.transform(volume, fill_value=np.nan)
            plt.pcolormesh(flat_view, zorder=1, cmap=cmap_view, vmin=0, vmax=vmax)

            # injection location
            pixel = np.ma.masked_where(mapper.view_lookup != val, flat_view, copy=False)
            plt.pcolormesh(pixel, zorder=2, cmap=cmap_pixel, vmin=0, vmax=1)

            # plot params
            plt.gca().invert_yaxis()
            plt.axis('off')
            plt.savefig(os.path.join(OUTPUT_DIR, '%05d.png' % i),
                        bbox_inches='tight', facecolor=None, edgecolor=None,
                        transparent=True, dpi=50)
            plt.close()

    logging.info('converting images to gif')
    subprocess.run(GIF_CONVERT_COMMAND, stdout=subprocess.PIPE, cwd=OUTPUT_DIR, shell=True)

if __name__ == '__main__':
    # default logging
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    main()
