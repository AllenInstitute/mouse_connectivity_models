from __future__ import division
import os
import logging
import nrrd
import numpy as np

# converts to double lol yeah right
#from skimage.transform import resize
from scipy.ndimage.interpolation import zoom
import allensdk.core.json_utilities as ju

from mcmodels.core import Mask, VoxelModelCache

from .py_neighborhood import masked_filter

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
TOP_DIR = os.path.join(FILE_DIR, '..', '..', '..')

INPUT_JSON = os.path.join(TOP_DIR, 'input.json')
OUTPUT_DIR = os.path.join(FILE_DIR, '..', 'output')
VOLUME_DIR = os.path.join(OUTPUT_DIR, 'volumes')

FULL_INJECTION = True
SCALE = 10
UPSCALE_KWARGS = dict(order=3, mode='nearest', prefilter=True)


def get_region_id(cache, acronym):
    return cache.get_structure_tree().get_structures_by_acronym([acronym])[0]['id']


def get_cortical_structure_ids(cache):
    st = cache.get_structure_tree()

    structure_ids = []
    for ss in st.get_structures_by_set_id([687527945]):
        if st.structure_descends_from(ss['id'], 315):
            structure_ids.append(ss['id'])

    return structure_ids


def filter_projection(cache, cortical_projection):

    # TODO: bilateral for now, should improve in future
    cortical_mask = Mask.from_cache(cache, structure_ids=[315], hemisphere_id=3)

    cortical_structures = get_cortical_structure_ids(cache)

    result = np.zeros_like(cortical_projection)
    for sid in cortical_structures:
        logging.debug("Filtering target %d", sid)

        left_mask = cache.get_structure_mask(sid)[0]
        right_mask = left_mask.copy()

        left_mask[..., left_mask.shape[-1]//2:] = 0
        right_mask[..., :right_mask.shape[-1]//2] = 0

        for mask, hid in zip((right_mask, left_mask), (2, 1)):
            key = cortical_mask.get_structure_indices(structure_ids=[sid],
                                                      hemisphere_id=hid)
            result += masked_filter(
                cortical_projection, mask, key=key, kind='median', axis=1,
                bounds=((-2, 2), (-2, 2), (-2, 2)))

    return result


def get_projection(cache, region_id, full=False, filtered=False):

    def get_centroid_voxel():
        region_mask = cache.get_structure_mask(region_id)[0]
        region_mask[..., :region_mask.shape[2]//2] = 0 # ipsi

        return np.argwhere(region_mask).mean(axis=0).astype(int)

    # get voxel array
    voxel_array, source_mask, target_mask = cache.get_voxel_connectivity_array()

    # get cortical targets
    col_idx = target_mask.get_structure_indices(structure_ids=[315], hemisphere_id=3)

    if full:
        logging.debug("Filling region")
        row_idx = source_mask.get_structure_indices(structure_ids=[region_id])
        projection = voxel_array[row_idx, col_idx]

        if filtered:
            logging.debug("Filtering target")
            projection = filter_projection(cache, projection)

        projection = projection.sum(axis=0) # we are normalizing anyway
    else:

        # single voxel in center
        logging.debug("Computing region centriod")
        row_idx = source_mask.get_flattened_voxel_index(get_centroid_voxel())
        projection = voxel_array[row_idx, col_idx]

        if filtered:
            logging.error("filtered keyword only implemented for full region")


    # scale to be in [0,1]
    projection /= projection.max()#6e-2
    projection.clip(min=0, max=1, out=projection)

    cortical_mask = Mask.from_cache(cache, structure_ids=[315], hemisphere_id=3)
    return cortical_mask.map_masked_to_annotation(projection)


def upscale_projection(projection, scale, **kwargs):
    arr = projection.astype(np.float32)
    del projection
    return zoom(arr, scale, output=np.float32, **kwargs).clip(0.0, 1.0)


def main(injection_region, filtered=False):
    input_data = ju.read(INPUT_JSON)

    manifest_file = input_data.get('manifest_file')
    manifest_file = os.path.join(TOP_DIR, manifest_file)

    log_level = input_data.get('log_level', logging.DEBUG)
    logging.getLogger().setLevel(log_level)

    # get voxel_model_cache
    cache = VoxelModelCache(manifest_file=manifest_file)

    # get region id
    region_id = get_region_id(cache, injection_region)

    logging.debug("performing virtual injection into %s (%s)" %
                  (injection_region, region_id))
    projection = get_projection(
        cache, region_id, full=FULL_INJECTION, filtered=filtered)

    # get projection (row)
    logging.debug("upscaling projection to 10 micron")
    projection = upscale_projection(projection, SCALE, **UPSCALE_KWARGS)

    # file name
    suffix = injection_region + "full" if FULL_INJECTION else injection_region
    vol_file = os.path.join(VOLUME_DIR, "projection_density_%s.nrrd" % suffix)
    logging.debug("saving projection volume : %s" % vol_file)
    nrrd.write(vol_file, projection, options=dict(encoding='raw'))

    return vol_file
