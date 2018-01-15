# Authors: Joseph Knox josephk@alleninstitute.org
# License:

def get_experiment_ids(mcc, structure_ids):
    """Returns all experiment ids given some structure_ids"""
    experiments = mcc.get_experiments(dataframe=True,
                                      injection_structure_ids=structure_ids)
    return experiments.injection_structure_ids


def get_id_acronym_map(mcc):
    """Returns dict id : acronym"""
    acronym_map = mcc.get_structure_tree().value_map(lambda x: x['id'],
                                                     lambda x: x['acronym'])
    return acronym_map
