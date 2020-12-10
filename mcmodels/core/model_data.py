# import numpy as np
# from mcmodels.core import VoxelData, RegionalData
# from mcmodels.utils import unionize
#
#
# class ModelData(object):
#
#     def __init__(self, cache, structure_id):
#         self.cache = cache
#         self.structure_id = structure_id
#
#     def get_structure_id(self, acronym):
#         structure_tree = self.cache.get_structure_tree()
#         return structure_tree.get_structures_by_acronym([acronym])[0]['id']
#
#     def get_experiment_ids(self, eid_set=None, experiments_exclude=[]):
#         """gets model data from ..."""
#
#         # get experiments
#         experiments = self.cache.get_experiments(
#             injection_structure_ids=[self.structure_id], cre=False)
#         experiment_ids = [e['id'] for e in experiments]
#
#         # exclude bad, restrict to eid_set
#         eid_set = experiment_ids if eid_set is None else eid_set
#         return set(experiment_ids) & set(eid_set) - set(experiments_exclude)
#
#     def get_voxel_data(self, **kwargs):
#         experiment_ids = self.get_experiment_ids(**kwargs)
#
#         data = VoxelData(self.cache, injection_structure_ids=[self.structure_id],
#                          injection_hemisphere_id=2)
#         data.get_experiment_data(experiment_ids)
#
#         return data
#
#     def get_regional_data(self, high_res=False, threshold_injection=True, **kwargs):
#         def get_summary_structure_ids():
#             structure_tree = self.cache.get_structure_tree()
#             structures = structure_tree.get_structures_by_set_id([687527945])
#
#             return [s['id'] for s in structures if s['id'] not in (934, 1009)]
#
#         def get_injection_regions(region_set):
#             """Return regions in region_set if descend from structure_id"""
#             st = self.cache.get_structure_tree()
#             return [r for r in region_set
#                     if st.structure_descends_from(r, self.structure_id)]
#         projection_hemisphere_id = kwargs.pop('projection_hemisphere_id', 3)
#
#         # get summary structures
#         region_set = get_summary_structure_ids()
#         injection_set = get_injection_regions(region_set)
#
#         # get experiments
#         experiment_ids = self.get_experiment_ids(**kwargs)
#
#         container = RegionalData if high_res else VoxelData
#         container_kwargs = dict(injection_structure_ids=injection_set,
#                                 projection_structure_ids=region_set,
#                                 injection_hemisphere_id=2,
#                                 projection_hemisphere_id=projection_hemisphere_id,
#                                 normalized_injection=True,
#                                 normalized_projection=True,
#                                 flip_experiments=True)
#
#         data = container(self.cache, **container_kwargs)
#         data.get_experiment_data(experiment_ids)
#
#         # get model data
#         if not high_res:
#             data.injections = unionize(
#                 data.injections, data.injection_mask.get_key(
#                     structure_ids=injection_set, hemisphere_id=2))
#             data.projections = unionize(
#                 data.projections, data.projection_mask.get_key(
#                     structure_ids=region_set, hemisphere_id=projection_hemisphere_id))
#
#         # threshold injection
#         # NOTE
#         if threshold_injection:
#             pct = np.percentile(data.injections[data.injections.nonzero()], 5)
#             data.injections[data.injections < pct] = 0
#
#         return data

#from mcmodels.core import VoxelData
from .base import VoxelData#, RegionalData
from mcmodels.utils import unionize #nonzero_unique,
import numpy as np

class ModelData(object):

    def __init__(self, cache, structure_id, structure_set_id = 687527945):
        self.cache = cache
        self.structure_id = structure_id
        self.structure_set_id = structure_set_id

    def get_structure_id(self, acronym):
        structure_tree = self.cache.get_structure_tree()
        return structure_tree.get_structures_by_acronym([acronym])[0]['id']

    def get_experiment_ids(self, eid_set=None, experiments_exclude=[], cre=None):
        """gets model data from ..."""

        # get experiments
        experiments = self.cache.get_experiments(
            # injection_structure_ids=[self.structure_id], cre=False)
            injection_structure_ids=[self.structure_id], cre=cre)
        experiment_ids = [e['id'] for e in experiments]

        # exclude bad, restrict to eid_set
        eid_set = experiment_ids if eid_set is None else eid_set
        return set(experiment_ids) & set(eid_set) - set(experiments_exclude)

    def get_voxel_data(self, **kwargs):
        print('here')
        # eid_set = kwargs.pop('eid_set')
        experiments_exclude = kwargs.pop('experiments_exclude')
        injection_hemisphere_id = kwargs.pop('injection_hemisphere_id')
        print(injection_hemisphere_id)
        cre = kwargs.pop('cre')
        experiment_ids = self.get_experiment_ids(experiments_exclude=experiments_exclude, cre=cre)
        # projection_hemisphere_id = kwargs.pop('projection_hemisphere_id'3)
        data = VoxelData(self.cache, injection_structure_ids=[self.structure_id],
                         # injection_hemisphere_id=2, flip_experiments = True)
                         injection_hemisphere_id=injection_hemisphere_id, flip_experiments=True)  # 2 + flip flips 1s into 2s
        data.get_experiment_data(experiment_ids)
        data.experiment_ids = experiment_ids
        return data

    def get_regional_data_fromvoxel(self, **kwargs):
        data = self.get_voxel_data(**kwargs)

    def get_regional_data(self, high_res=False, threshold_injection=True, **kwargs):
        def get_summary_structure_ids():
            structure_tree = self.cache.get_structure_tree()
            structures = structure_tree.get_structures_by_set_id([self.structure_set_id])

            return [s['id'] for s in structures if s['id'] not in (934, 1009)]

        def get_injection_regions(region_set):
            """Return regions in region_set if descend from structure_id"""
            st = self.cache.get_structure_tree()
            return [r for r in region_set
                    if st.structure_descends_from(r, self.structure_id)]

        projection_hemisphere_id = kwargs.pop('projection_hemisphere_id', 3)

        # get summary structures
        region_set = get_summary_structure_ids()
        injection_set = get_injection_regions(region_set)

        # get experiments
        experiment_ids = self.get_experiment_ids(**kwargs)

        container = RegionalData if high_res else VoxelData
        container_kwargs = dict(injection_structure_ids=injection_set,
                                projection_structure_ids=region_set,
                                injection_hemisphere_id=2,
                                projection_hemisphere_id=projection_hemisphere_id,
                                normalized_injection=True,
                                normalized_projection=True,
                                flip_experiments=True)

        data = container(self.cache, **container_kwargs)
        data.get_experiment_data(experiment_ids)

        # get model data
        if not high_res:
            data.injections = unionize(
                data.injections, data.injection_mask.get_key(
                    structure_ids=injection_set, hemisphere_id=2))
            data.projections = unionize(
                data.projections, data.projection_mask.get_key(
                    structure_ids=region_set, hemisphere_id=projection_hemisphere_id))

        # threshold injection
        # NOTE
        if threshold_injection:
            pct = np.percentile(data.injections[data.injections.nonzero()], 5)
            data.injections[data.injections < pct] = 0

        return data
