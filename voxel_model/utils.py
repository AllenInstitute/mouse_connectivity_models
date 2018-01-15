# Authors: Joseph Knox josephk@alleninstitute.org
# License:

def get_experiment_ids(mcc, structure_ids, cre=None):
    """Returns all experiment ids given some structure_ids
    PRIMARY INJECTION STRUCTURES
    """
    # filters injections by structure id OR DECENDENT
    experiments = mcc.get_experiments(dataframe=False, cre=cre,
                                      injection_structure_ids=structure_ids)
    return [ experiment['id'] for experiment in experiments ]


def get_id_acronym_map(mcc):
    """Returns dict id : acronym"""
    acronym_map = mcc.get_structure_tree().value_map(lambda x: x['id'],
                                                     lambda x: x['acronym'])
    return acronym_map
