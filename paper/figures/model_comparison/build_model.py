from __future__ import division
import os
import logging

import numpy as np
import pandas as pd
import allensdk.core.json_utilities as ju

from mcmodels.core import VoxelModelCache, Mask
from mcmodels.models.voxel import RegionalizedModel
from mcmodels.regressors.nonparametric.kernels import Polynomial
from mcmodels.utils import padded_diagonal_fill

from helpers.model_data import ModelData
from helpers.error import VoxelModelError
from helpers.utils import get_structure_id, get_ordered_summary_structures

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
TOP_DIR = os.path.join(FILE_DIR, '..', '..')
OUTPUT_DIR = os.path.join(FILE_DIR, 'output')

INPUT_JSON = os.path.join(TOP_DIR, 'input.json')
EXPERIMENTS_EXCLUDE_JSON = os.path.join(TOP_DIR, 'experiments_exclude.json')

LOG = True


def fit_structure(cache, structure_id, experiments_exclude, kernel_params,
                  model_option='standard'):
    data = ModelData(cache, structure_id).get_voxel_data(
        experiments_exclude=experiments_exclude)

    # nested cross val
    nw_kwargs = dict()
    if 'shape' in kernel_params:
        nw_kwargs['kernel'] = Polynomial(**kernel_params)
    else:
        nw_kwargs['kernel'] = 'rbf'
        nw_kwargs['gamma'] = kernel_params.pop('gamma')


    error = VoxelModelError(cache, data)
    return data, error.fit(**nw_kwargs, option=model_option)


def main():
    input_data = ju.read(INPUT_JSON)

    structures = input_data.get('structures')
    manifest_file = input_data.get('manifest_file')
    manifest_file = os.path.join(TOP_DIR, manifest_file)

    log_level = input_data.get('log_level', logging.DEBUG)
    logging.getLogger().setLevel(log_level)

    # experiments to exclude
    experiments_exclude = ju.read(EXPERIMENTS_EXCLUDE_JSON)

    # load hyperparameter dict
    suffix = 'log' if LOG else 'standard'
    hyperparameter_json = os.path.join(OUTPUT_DIR, 'hyperparameters-%s.json' % suffix)
    hyperparameters = ju.read(hyperparameter_json)

    # get caching object
    cache = VoxelModelCache(manifest_file=manifest_file)

    # get structure ids
    structure_ids = [get_structure_id(cache, s) for s in structures]

    # mask for reordering source
    annotation = cache.get_annotation_volume()[0]
    cumm_source_mask = np.zeros(annotation.shape, dtype=np.int)

    offset = 1 # start @ 1 so that nonzero can be used
    weights, nodes = [], []
    for sid, sac in zip(structure_ids, structures):
        logging.debug("Building model for structure: %s", sac)

        data, reg = fit_structure(cache, sid, experiments_exclude,
                                  hyperparameters[sac], model_option=suffix)

        w = reg.get_weights(data.injection_mask.coordinates)

        # assign ordering to full source
        ordering = np.arange(offset, w.shape[0] + offset, dtype=np.int)
        offset += w.shape[0]

        # get source mask
        data.injection_mask.fill_volume_where_masked(cumm_source_mask, ordering)

        # append to list
        weights.append(w)
        nodes.append(reg.nodes)

    # stack
    weights = padded_diagonal_fill(weights)
    nodes = np.vstack(nodes)

    # need to reorder weights
    # (subtract 1 to get proper index)
    permutation = cumm_source_mask[cumm_source_mask.nonzero()] - 1
    weights = weights[permutation, :]

    # regionalized
    logging.debug('regionalizing voxel weights')
    ontological_order = get_ordered_summary_structures(cache)
    source_mask = Mask.from_cache(cache, structure_ids=structure_ids, hemisphere_id=2)
    source_key = source_mask.get_key(structure_ids=ontological_order)
    ipsi_key = data.projection_mask.get_key(structure_ids=ontological_order, hemisphere_id=2)
    contra_key = data.projection_mask.get_key(structure_ids=ontological_order, hemisphere_id=1)
    ipsi_model = RegionalizedModel(weights, nodes, source_key, ipsi_key,
                                     ordering=ontological_order, dataframe=True)
    contra_model = RegionalizedModel(weights, nodes, source_key, contra_key,
                                     ordering=ontological_order, dataframe=True)
    get_metric = lambda s: pd.concat((getattr(ipsi_model, s), getattr(contra_model, s)),
                                     keys=('ipsi', 'contra'), axis=1)

    # write results
    output_dir = os.path.join(TOP_DIR, 'connectivity', 'voxel-%s-model' % suffix)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # regionalized
    logging.debug('saving to directory: %s', output_dir)
    get_metric('connection_density').to_csv(
        os.path.join(output_dir, 'connection_density.csv'))
    get_metric('connection_strength').to_csv(
        os.path.join(output_dir, 'connection_strength.csv'))
    get_metric('normalized_connection_density').to_csv(
        os.path.join(output_dir, 'normalized_connection_density.csv'))
    get_metric('normalized_connection_strength').to_csv(
        os.path.join(output_dir, 'normalized_connection_strength.csv'))

    # voxel
    ju.write(os.path.join(output_dir, 'target_mask_params.json'),
             dict(structure_ids=structure_ids, hemisphere_id=3))
    ju.write(os.path.join(output_dir, 'source_mask_params.json'),
             dict(structure_ids=structure_ids, hemisphere_id=2))
    np.savetxt(os.path.join(output_dir, 'weights.csv.gz'),
               weights.astype(np.float32), delimiter=',')
    np.savetxt(os.path.join(output_dir, 'nodes.csv.gz'),
               nodes.astype(np.float32), delimiter=',')


if __name__ == "__main__":
    main()
