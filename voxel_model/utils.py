# Authors: Joseph Knox josephk@alleninstitute.org
# License:

from __future__ import absolute_import
import numpy as np
from .experiment import Experiment

def get_experiment_ids(mcc, structure_ids, cre=None):
    """Returns all experiment ids given some structure_ids
    PRIMARY INJECTION STRUCTURES
    """
    # filters injections by structure id OR DECENDENT
    experiments = mcc.get_experiments(dataframe=False, cre=cre,
                                      injection_structure_ids=structure_ids)
    return [ experiment['id'] for experiment in experiments ]

def get_model_data(mcc, experiment_ids, source_mask, target_mask):
    """..."""
    injections=[]
    projections=[]
    centroids=[]

    for eid in experiment_ids:
        # pull experimental data
        experiment = Experiment(mcc, eid, normalize_projection=True)

        # NEED TO MASK
        centroid = experiment.centroid
        injection = experiment.injection_density[source_mask.where]
        projection = experiment.projection_density[target_mask.where]

        # update
        injections.append(injection)
        projections.append(projection)
        centroids.append(centroid)

    # return arrays
    source_voxels = source_mask.coordinates
    X = np.hstack( (np.asarray(centroids), np.asarray(injections)) )
    y = np.asarray( projections )

    return source_voxels, X, y

def get_id_acronym_map(mcc):
    """Returns dict id : acronym"""
    acronym_map = mcc.get_structure_tree().value_map(lambda x: x['id'],
                                                     lambda x: x['acronym'])
    return acronym_map
