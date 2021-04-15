import numpy as np
from .model_data import ModelData
from .structure_data import StructureData
from .utils import get_ccf_data
from .utils import get_masked_data_volume
from .mask import Mask
from .utils import nonzero_unique, unionize
from sklearn.metrics import pairwise_distances

def get_minorstructures(eids, data_info, ai_map):
    experiments_minors = np.zeros(len(eids), dtype=object)
    for i in range(len(eids)):
        experiment_id = eids[i]
        experiments_minors[i] = ai_map[data_info['primary-injection-structure'].loc[experiment_id]]
    return (experiments_minors)

def get_connectivity_data(cache, structure_ids, experiments_exclude, remove_injection=False, structure_set_id = 687527945):
    '''

    :param cache: VoxelModelCache for communicating with AllenSDK
    :param structure_ids: Ids for which to pull data
    :param experiments_exclude: Experiments to exclude (not tested)
    :param remove_injection: Remove injection signal from projection (not tested)
    :param structure_set_id: Structure set specifying list of structures (not same as location of structures)
    :return:
    '''
    connectivity_data = ConnectivityData(cache)

    for sid in structure_ids:
        print(sid)
        sid_data = StructureData(sid)
        # deprecated language
        model_data = ModelData(cache, sid, structure_set_id)
        sid_data.eids = model_data.get_experiment_ids(experiments_exclude=experiments_exclude, cre=None)
        for eid in sid_data.eids:

            eid_data = get_ccf_data(cache, eid)  # ExperimentData(eid)
            eid_data.data_mask_tolerance = .5
            # ccf_data = get_ccf_data(cache, eid)
            eid_data.injection_signal_true = eid_data.injection_signal * eid_data.injection_fraction
            if remove_injection == True:
                pass  # remove injection fraction from projection
            # injection_signal should = projection_signal in some locations (nonzero)
            # why do we use partial?
            # mask_func = partial(_mask_data_volume,data_mask=eid_data.data_mask,tolerance=eid_data.data_mask_tolerance)
            eid_data.injection_qmasked = get_masked_data_volume(eid_data.injection_signal_true, eid_data.data_quality_mask,
                                                           eid_data.data_mask_tolerance)
            eid_data.projection_qmasked = get_masked_data_volume(eid_data.projection_signal, eid_data.data_quality_mask,
                                                            eid_data.data_mask_tolerance)  # mask_func(eid_data.projection_signal)
            # eid_data.centroid = compute_centroid(eid_data.injection_qmasked)
            sid_data.experiment_datas[eid] = eid_data
        connectivity_data.structure_datas[sid] = sid_data
    return (connectivity_data)

class ConnectivityData():
    '''
    Basic experimental class for holding data from multiple major structures (e.g. CB)
    Rewrites much of the functionality from Knox.
    The goal is to more explicitly state the algorithmic steps.
    '''
    def __init__(self,cache):
        self.structure_datas = {}
        self.cache = cache

    def get_injection_hemisphere_ids(self):

        structure_datas = self.structure_datas

        for sid in list(structure_datas.keys()):
            structure_datas[sid].get_injection_hemisphere_ids()

        self.structure_datas = structure_datas

    def align(self):
        '''
        Align all experiments to have same hemisphere
        :return: self
        '''
        structure_datas = self.structure_datas

        for sid in list(structure_datas.keys()):
            structure_datas[sid].align()
        self.structure_datas = structure_datas

    def get_centroids(self):
        '''
        Compute centroids of all experiments in dataset
        :return: self
        '''
        structure_datas = self.structure_datas

        for sid in list(structure_datas.keys()):
            structure_datas[sid].get_centroids()

        self.structure_datas = structure_datas


    def get_data_matrices(self,default_structure_ids):

        connectivity_data = self
        cache = self.cache
        #default_structure_ids = self.default_structure_ids

        structure_ids = np.asarray(list(connectivity_data.structure_datas.keys()))
        for sid in structure_ids:
            experiment_ids = np.asarray(list(connectivity_data.structure_datas[sid].experiment_datas.keys()))
            connectivity_data.structure_datas[sid].injection_mask = Mask.from_cache(cache, structure_ids=[sid],
                                                                                    hemisphere_id=2)
            connectivity_data.structure_datas[sid].projection_mask = Mask.from_cache(cache,
                                                                                     structure_ids=default_structure_ids,
                                                                                     hemisphere_id=3)
            for eid in experiment_ids:
                connectivity_data.structure_datas[sid].experiment_datas[eid].injection_vec = \
                connectivity_data.structure_datas[sid].injection_mask.mask_volume(
                    connectivity_data.structure_datas[sid].experiment_datas[eid].injection_qmasked)
                connectivity_data.structure_datas[sid].experiment_datas[eid].projection_vec = \
                connectivity_data.structure_datas[sid].projection_mask.mask_volume(
                    connectivity_data.structure_datas[sid].experiment_datas[eid].projection_qmasked)
            connectivity_data.structure_datas[sid].injections = np.asarray(
                [connectivity_data.structure_datas[sid].experiment_datas[eid].injection_vec for eid in
                 connectivity_data.structure_datas[sid].eids])
            connectivity_data.structure_datas[sid].projections = np.asarray(
                [connectivity_data.structure_datas[sid].experiment_datas[eid].projection_vec for eid in
                 connectivity_data.structure_datas[sid].eids])
            connectivity_data.structure_datas[sid].centroids = np.asarray(
                [connectivity_data.structure_datas[sid].experiment_datas[eid].centroid for eid in
                 connectivity_data.structure_datas[sid].eids])

        #return (connectivity_data)

    def get_crelines(self,data_info):
        connectivity_data= self
        major_structure_ids = np.asarray(list(connectivity_data.structure_datas.keys()))
        exps = np.asarray(data_info.index.values , dtype = np.int)
        creline = {}
        for sid in major_structure_ids:
            experiment_ids = np.asarray(list(connectivity_data.structure_datas[sid].experiment_datas.keys()))
            nexp = len(experiment_ids)
            creline[sid] = np.zeros(nexp, dtype = object)
            for i in range(len(experiment_ids)):
                index = np.where(exps == experiment_ids[i])[0][0]
                creline[sid][i] = data_info['transgenic-line'].iloc[index]
        self.creline = creline

    def get_summarystructures(self, data_info):

        connectivity_data = self
        ai_map = self.ai_map

        summarystructure_dictionary = {}
        major_structure_ids = np.asarray(list(connectivity_data.structure_datas.keys()))
        for sid in major_structure_ids:
            eids = np.asarray(list(connectivity_data.structure_datas[sid].experiment_datas.keys()))
            connectivity_data.structure_datas[sid].summary_structures = get_minorstructures(eids, data_info,ai_map)
        #return (connectivity_data)

    def get_leafs(self, leafs):
        connectivity_data = self

        #summarystructure_dictionary = {}
        major_structure_ids = np.asarray(list(connectivity_data.structure_datas.keys()))
        for sid in major_structure_ids:
            #eids = np.asarray(list(connectivity_data.structure_datas[sid].experiment_datas.keys()))
            connectivity_data.structure_datas[sid].leafs = leafs[sid]
        #return (connectivity_data)


    def get_regionalized_normalized_data(self, source_order, ipsi_key,
                                         contra_key):  # experiments_minor_structures):
        '''
        :param msvds: Class dictionary holding data
        :param cache: AllenSDK cache
        :param source_order: Source key (tautologically ipsilateral due to hemisphere mirroring)
        :param ipsi_key: Ipsilateral target key
        :param contra_key:  Contralateral target key
        :return: reg_proj: the regionalized projection vector
                reg_proj_vcount_norm: the regionalized projection normalized by target size
                 reg_proj_norm: the regionalized normalized projection vector
                   reg_inj_vcount_norm: the regionalized injection normalized by source size
                    reg_inj: the regionalized injection vector
                    reg_proj_vcount_norm_injnorm:  this doesn't make sense (divise target size normalized projection by sum of source size normalized injection)
                    reg_proj_injnorm: the regionalized projection vector normalized by total injection (the one from Knox et al)

        '''
        connectivity_data = self
        cache = self.cache

        major_structure_ids = np.asarray(list(connectivity_data.structure_datas.keys()))
        for sid in major_structure_ids:
            # print()
            structure_data = connectivity_data.structure_datas[sid]
            # nexp = msvd.projections.shape[0]

            # minor_structures = np.unique(experiments_minor_structures[sid])
            # nmins = len(minor_structures)

            projections = structure_data.projections
            ipsi_proj = unionize(projections, ipsi_key)
            contra_proj = unionize(projections, contra_key)
            reg_proj = np.hstack([ipsi_proj, contra_proj])
            structure_data.reg_proj = reg_proj

            ipsi_target_regions, ipsi_target_counts = nonzero_unique(ipsi_key, return_counts=True)
            contra_target_regions, contra_target_counts = nonzero_unique(contra_key, return_counts=True)
            target_counts = np.concatenate([ipsi_target_counts, contra_target_counts])
            reg_proj_vcount_norm = np.divide(reg_proj, target_counts[np.newaxis, :])
            structure_data.reg_proj_vcount_norm = reg_proj_vcount_norm
            structure_data.reg_proj_vcount_norm_renorm = reg_proj_vcount_norm / np.expand_dims(
                #np.linalg.norm(reg_proj_vcount_norm, axis=1), 1)
                np.sum(reg_proj_vcount_norm, axis=1), 1)
            structure_data.reg_proj_norm = reg_proj / np.expand_dims(np.sum(reg_proj, axis=1), 1)#np.expand_dims(np.linalg.norm(reg_proj, axis=1), 1)

            source_mask = Mask.from_cache(cache, structure_ids=[sid], hemisphere_id=2)
            source_key = source_mask.get_key(structure_ids=source_order)
            source_regions, source_counts = nonzero_unique(source_key, return_counts=True)

            injections = structure_data.injections
            reg_ipsi_inj = unionize(injections, source_key)
            structure_data.reg_inj = reg_ipsi_inj
            reg_inj_vcount_norm = np.divide(reg_ipsi_inj, source_counts[np.newaxis, :])
            structure_data.reg_inj_vcount_norm = reg_inj_vcount_norm

            structure_data.reg_proj_vcount_norm_injnorm = reg_proj_vcount_norm / np.expand_dims(
                #np.linalg.norm(reg_inj_vcount_norm, axis=1), 1)
                np.sum(reg_inj_vcount_norm, axis=1), 1)

            reg_proj_injnorm = reg_proj / np.expand_dims(
                #np.linalg.norm(reg_ipsi_inj, axis=1), 1)
                np.sum(reg_ipsi_inj, axis=1), 1)

            structure_data.reg_proj_injnorm = reg_proj_injnorm
            connectivity_data.structure_datas[sid] = structure_data
            # msvd.reg_proj_vcountnorm_totalnorm =
        connectivity_data.ipsi_target_regions = ipsi_target_regions
        connectivity_data.contra_target_regions = contra_target_regions
        connectivity_data.target_regions = np.concatenate([ipsi_target_regions, contra_target_regions])
        #return (connectivity_data)


    def get_creleaf_combos(self):

        connectivity_data = self
        leafs = self.leafs
        creline = self.creline
        creleafs = {}
        creleafs_merged = {}

        major_structure_ids = np.asarray(list(connectivity_data.structure_datas.keys()))
        for sid in major_structure_ids:
            creleafs[sid] = np.asarray(np.vstack([leafs[sid], creline[sid]]), dtype=str).transpose()
            creleafs_merged[sid] = [creleafs[sid][:, 0][i] + creleafs[sid][:, 1][i] for i in range(creleafs[sid].shape[0])]
            creleafs_merged[sid] = np.asarray(creleafs_merged[sid])
        self.creleaf_combos = creleafs_merged

    def get_pairwise_distances(self):
        connectivity_data = self
        major_structure_ids = np.asarray(list(connectivity_data.structure_datas.keys()))
        for sid in major_structure_ids:
            connectivity_data.structure_datas[sid].pairwise_distances = pairwise_distances(connectivity_data.structure_datas[sid].centroids)**2

    #
    def get_summarystructure_major_dictionary(self):
        connectivity_data = self
        structure_major_dictionary = {}
        keys = np.asarray(list(connectivity_data.structure_datas))
        for sid in keys:
            strs_sid = np.unique(connectivity_data.structure_datas[sid].summary_structures)
            nstrs = len(strs_sid)
            for s in range(nstrs):
                structure_major_dictionary[strs_sid[s]] = sid
        self.structure_major_dictionary = structure_major_dictionary
        #return (structure_major_dictionary)


    def get_major_summarystructure_dictionary(self):
        connectivity_data = self
        major_structure_dictionary = {}
        keys = np.asarray(list(connectivity_data.structure_datas))
        for sid in keys:
            strs_sid = np.unique(connectivity_data.structure_datas[sid].summary_structures)
            major_structure_dictionary[sid] = np.asarray(strs_sid, dtype=int)
        self.major_summarystructure_dictionary = major_structure_dictionary
        #return (major_structure_dictionary)


    def get_leaf_major_dictionary(self):
        connectivity_data = self
        structure_major_dictionary = {}
        keys = np.asarray(list(connectivity_data.structure_datas))
        for sid in keys:
            strs_sid = np.unique(connectivity_data.leafs[sid])
            nstrs = len(strs_sid)
            for s in range(nstrs):
                structure_major_dictionary[strs_sid[s]] = sid#connectivity_data.leafs[sid]
        self.leaf_major_dictionary = structure_major_dictionary
        #return (structure_major_dictionary)


    def get_major_leaf_dictionary(self):
        connectivity_data = self
        major_structure_dictionary = {}
        keys = np.asarray(list(connectivity_data.structure_datas))
        for sid in keys:
            strs_sid = np.unique(connectivity_data.leafs[sid])
            major_structure_dictionary[sid] = np.asarray(strs_sid, dtype=int)
        self.major_leaf_dictionary = major_structure_dictionary
        #return (major_structure_dictionary)

    def get_cresum_combos(self):

        connectivity_data = self
        summary_structures = self.summary_structures
        creline = self.creline
        cresums = {}
        cresums_merged = {}

        major_structure_ids = np.asarray(list(connectivity_data.structure_datas.keys()))
        for sid in major_structure_ids:
            cresums[sid] = np.asarray(np.vstack([summary_structures[sid], creline[sid]]), dtype=str).transpose()
            cresums_merged[sid] = [cresums[sid][:, 0][i] + cresums[sid][:, 1][i] for i in range(cresums[sid].shape[0])]
            cresums_merged[sid] = np.asarray(cresums_merged[sid])
        self.cresum_combos = cresums_merged
