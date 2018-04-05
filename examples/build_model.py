# Authors:
# License:

from __future__ import print_function
import os
import json
import numpy as np

from sklearn.gaussian_process.kernels import Matern

from mcmodels.masks import Mask
from mcmodels.model_data import ModelData, get_experiment_ids
from mcmodels.regressors import InjectionModel
from mcmodels.utils import get_mcc, padded_diagonal_fill

def main(structure_ids, manifest_file, experiment_exclude_file,
         parameter_file, output_dir):

    # grab MouseConnectivityCache instance
    mcc = get_mcc(manifest_file)

    # get set of experiment ids to exclude
    with open(experiment_exclude_file) as fn:
        experiment_exclude = set(json.load(fn))

    # get parameters
    with open(parameter_file) as fn:
        parameter_dict = json.load(fn)

    # all target masks the same
    target_mask = Mask(mcc, hemisphere=3)

    # mask for reordering source
    offset = 1  # start @ 1 so that nonzero can be used
    cumm_source_mask = np.zeros(target_mask.annotation_shape, dtype=np.int)

    weights = []
    nodes = []
    for sid in structure_ids:
        print("building model for", sid)

        # get source mask
        source_mask = Mask(mcc, structure_ids=[sid], hemisphere=2)

        # get experiments (only for wild type
        experiment_ids = get_experiment_ids(mcc, [sid], cre=False)
        experiment_ids = set(experiment_ids) - experiment_exclude

        # get model data
        model_data = ModelData.from_mcc_and_masks(mcc, source_mask, target_mask,
                                                  experiment_ids=experiment_ids)

        # get hyperparameters from hyperparameter fitting
        parameters = parameter_dict[str(sid)]  # json serialized
        kernel = Matern(**parameters)

        # build model
        voxel_model = InjectionModel(model_data.source_voxels, kernel=kernel)
        voxel_model.fit((model_data.centroids, model_data.injections),
                        model_data.projections)

        # append to lists
        weights.append(voxel_model.weights)
        nodes.append(voxel_model.nodes)

        # assign ordering to full source
        n_rows = voxel_model.weights.shape[0]
        ordering = np.arange(offset, n_rows + offset, dtype=np.int)
        source_mask.fill_volume_where_masked(cumm_source_mask, ordering)

        # update offset
        offset += n_rows

    # data
    weights = padded_diagonal_fill(weights)
    nodes = np.vstack(nodes)

    # need to reorder weights
    # (subtract 1 to get proper index)
    permutation = cumm_source_mask[cumm_source_mask.nonzero()] - 1
    weights = weights[permutation, :]

    # get union of source masks
    full_source = Mask(mcc, hemisphere=2)

    print("saving")
    # save model
    np.save(os.path.join(output_dir, "weights.npy"), weights)
    np.save(os.path.join(output_dir, "nodes.npy"), nodes)

    # save masks
    full_source.save(os.path.join(output_dir, "source_mask.pkl"))
    target_mask.save(os.path.join(output_dir, "target_mask.pkl"))


if __name__ == "__main__":

    # model will be build for each disjoint structure
    STRUCTURE_IDS = [
        315,   # Isocortex
        698,   # OLF
        1089,  # HPF
        703,   # CTXsp
        477,   # STR
        803,   # PAL
        549,   # TH
        1097,  # HY
        313,   # MB
        771,   # P
        354,   # MY
        512    # CB
    ]

    # mcc settings
    MANIFEST_FILE = os.path.join(os.getcwd(), "connectivity",
                                 "mouse_connectivity_manifest.json")

    # i/o settings
    EXPERIMENT_EXCLUDE_FILE = os.path.join(os.path.dirname(__file__),
                                           "experiment_exclude.json")
    PARAMETER_FILE = os.path.join(os.path.dirname(__file__),
                                  "hyperparameters.json")
    OUTPUT_DIR = os.path.join(os.getcwd(), "data")

    # make sure dir exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    main(STRUCTURE_IDS, MANIFEST_FILE, EXPERIMENT_EXCLUDE_FILE,
         PARAMETER_FILE, OUTPUT_DIR)
