from .utils import get_centroid, get_injection_hemisphere_id

class StructureData():

    def __init__(self, sid):
        self.experiment_datas = {}

    def get_injection_hemisphere_ids(self):

        experiment_datas = self.experiment_datas

        for eid in list(experiment_datas.keys()):
            experiment_datas[eid].injection_hemisphere_id = get_injection_hemisphere_id(
                experiment_datas[eid].injection_qmasked, majority=True)
        self.experiment_datas = experiment_datas

    def align(self):

        experiment_datas = self.experiment_datas

        for eid in list(experiment_datas.keys()):
            if experiment_datas[eid].injection_hemisphere_id == 1:
                experiment_datas[eid].flip()
        self.experiment_datas = experiment_datas

    def get_centroids(self):

        experiment_datas = self.experiment_datas

        for eid in list(experiment_datas.keys()):
            experiment_datas[eid].centroid = get_centroid(experiment_datas[eid].injection_qmasked)
        self.experiment_datas = experiment_datas
