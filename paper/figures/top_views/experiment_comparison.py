from __future__ import division
import os
import json
import logging

import numpy as np
import allensdk.core.json_utilities as ju
from mcmodels.core import VoxelModelCache

from helpers.top_down_view import TopDownView
from helpers.smoothed_model import SmoothedModel, OptimizedSmoothedModel


FILE_DIR = os.path.dirname(os.path.realpath(__file__))
TOP_DIR = os.path.join(FILE_DIR, '..', '..')

INPUT_JSON = os.path.join(TOP_DIR, 'input.json')
OUTPUT_DIR = os.path.join(FILE_DIR, 'output')

EXPERIMENT_IDS = dict(Isocortex=(141602484, 115958825, 304565427, 112935169, 112514202, 174360333),
                      OLF      =(121509711, 131068390, 126653015),
                      HPF      =(100141214, 113166056, 142654100, 585051446),
                      CTXsp    =(125832322, 113144533),
                      PAL      =(117312486, 138058320, 125436508),
                      STR      =(146985623, 100141435),
                      TH       =(158375425, 266585624, 146046430, 113784293, 272873704, 156931568),
                      HY       =(112228391, 112372418),
                      MB       =(175263063, 127083591, 113399428, 100141993, 100141596),
                      P        =(272700063, 158434409, 121145045, 147633885, 156929391),
                      MY       =(159648854, 114402050, 127353220, 128056535, 272738620),
                      CB       =(112424102, 127398177, 125801739, 120493315))

BLEND_FACTOR = 0.6
CMAP_FILE = os.path.join(FILE_DIR, 'colormap.txt')

SMOOTHED = False
LOG = False
EPSILON = 1e-8


def main(experiment_ids):
    input_data = ju.read(INPUT_JSON)

    manifest_file = input_data.get('manifest_file')
    manifest_file = os.path.join(TOP_DIR, manifest_file)

    log_level = input_data.get('log_level', logging.DEBUG)
    logging.getLogger().setLevel(log_level)

    # get voxel_model_cache
    cache = VoxelModelCache(manifest_file=manifest_file)

    # get model weights
    logging.debug('loading model')
    voxel_array, source_mask, target_mask = cache.get_voxel_connectivity_array()

    # file save suffix
    if SMOOTHED:
        suffix = 'smoothed'
        if LOG:
            suffix += '-log'
    elif LOG:
        suffix = 'log'
    else:
        suffix = 'standard'


    # top down viewer
    tdv = TopDownView(cache, CMAP_FILE, blend_factor=BLEND_FACTOR)
    if SMOOTHED:
        osm = OptimizedSmoothedModel(cache, voxel_array, source_mask, target_mask)

    for ss, experiment_ids in experiment_ids.items():
        for eid in experiment_ids:
            # get experiment projection/centroid
            logging.debug('loading experiment %d', eid)
            experiment = cache.get_projection_density(eid)[0]

            logging.debug('getting model weights')
            # ---------
            # get model weights
            centroid = tdv.get_experiment_centroid(eid)

            if SMOOTHED:
                logging.debug("Filtering target")
                volume = osm.fit_voxel(centroid)
            else:
                row = source_mask.get_flattened_voxel_index(centroid)
                volume = voxel_array[row]

            if LOG:
                logging.debug("inverse log transforming target")
                volume = np.power(10.0, volume)
                volume[volume < EPSILON] = 0 # instead of x-EPS for numerical reasons

            model = target_mask.map_masked_to_annotation(volume)
            # ---------

            # get image
            logging.debug('creating images')
            exp = tdv.get_top_view(experiment)
            mod = tdv.get_top_view(model)

            # save
            output_dir = os.path.join(OUTPUT_DIR, ss, str(eid))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            logging.debug('saving')
            exp.save(os.path.join(output_dir, 'data_%d.png' % eid))
            mod.save(os.path.join(output_dir, '%s_model_%d.png' % (suffix, eid)))

            out_data = dict(experiment_id=eid, centroid=centroid)
            with open(os.path.join(output_dir, 'out_%d.json' % eid), 'w') as f:
                json.dump(out_data, f, indent=2)


if __name__ == '__main__':
    main(EXPERIMENT_IDS)
