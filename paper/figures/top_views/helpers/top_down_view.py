from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image


class TopDownView(object):

    ROOT_ID = 997

    def __init__(self, cache, cmap_file, blend_factor=0.5):
        self.cache = cache
        self.cmap_file = cmap_file
        self.blend_factor = blend_factor

    def _load_cmap(self):
        colors = []
        with open(self.cmap_file,'r') as f:
            for r in list(f):
                color = [float(c.strip())/255.0 for c in r.split()]
                colors.append(color)

        cmap = ListedColormap(colors)
        cmap.set_over([0.0, 0.0, 0.0])
        cmap.set_under([0.0, 0.0, 0.0])

        return cmap

    @property
    def cmap(self):
        try:
            return self._cmap
        except AttributeError:
            self._cmap = self._load_cmap()
            return self._cmap

    @property
    def root_mask(self):
        try:
            return self._root_mask
        except AttributeError:
            self._root_mask = self.cache.get_structure_mask(self.ROOT_ID)[0]
            return self._root_mask

    def get_experiment_centroid(self, experiment_id):
        injection_density = self.cache.get_injection_density(experiment_id)[0]
        injection_fraction = self.cache.get_injection_fraction(experiment_id)[0]

        centroid = self.cache.api.calculate_injection_centroid(
            injection_density, injection_fraction, resolution=1)

        return tuple(map(int, centroid))

    def get_top_view(self, volume):
        top_view = lambda arr: arr.sum(axis=1)
        image = lambda arr, func: Image.fromarray((np.uint8(func(arr)*255)))

        volume = top_view(volume)
        volume /= volume.max()
        volume = image(volume, self.cmap)

        mask = top_view(self.root_mask).astype(np.uint8)
        mask = mask * (1 / mask.max())
        mask[mask == 0] = 1
        mask = image(mask, lambda a: plt.cm.gray(a))

        return Image.blend(mask, volume, self.blend_factor)
