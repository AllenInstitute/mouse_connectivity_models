# Authors: Joseph Knox josephk@alleninstitute.org
# License: 

def get_id_acronym_map(mcc):
    """Returns dict id : acronym"""
    acronym_map = mcc.get_structure_tree().value_map(lambda x: x['id'], 
                                                       lambda x: x['acronym'])
