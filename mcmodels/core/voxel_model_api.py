"""
Module containing VoxelModelApi.
"""
# Authors: Joseph Knox <josephk@alleninstitute.org>
# License: Allen Institute Software License

from allensdk.api.cache import cacheable, Cache
from allensdk.api.queries.mouse_connectivity_api import MouseConnectivityApi


class VoxelModelApi(MouseConnectivityApi):
    '''HTTP Client extending MouseConnectivityApi to download model data.
    '''
    HTTP_MODEL_DIRECTORY = "http://download.alleninstitute.org/publications/"\
            "A_high_resolution_data-driven_model_of_the_mouse_connectome/"

    NODES_FILE = "C57BL-6J_nodes.npy"
    WEIGHTS_FILE = "C57BL-6J_weights.npy"
    SOURCE_MASK_FILE = "source_mask_params.json"
    TARGET_MASK_FILE = "target_mask_params.json"

    CONNECTION_DENSITY_FILE = 'connection_density.csv.gz'
    CONNECTION_STRENGTH_FILE = 'connection_strength.csv.gz'
    NORMALIZED_CONNECTION_DENSITY_FILE = 'normalized_connection_density.csv.gz'
    NORMALIZED_CONNECTION_STRENGTH_FILE = 'normalized_connection_strength.csv.gz'

    def download_model_files(self, file_name, save_file_path=None):
        """Download  data.

        Parameters
        ----------
        file_name : string, optional
        save_file_path : string, optional
            File name to save as.
        """
        url = self.HTTP_MODEL_DIRECTORY + file_name
        self.retrieve_file_over_http(url, save_file_path)

    @cacheable(strategy='create',
               pathfinder=Cache.pathfinder(file_name_position=1, path_keyword='path'))
    def download_nodes(self, file_name):
        self.download_model_files(self.NODES_FILE, file_name)

    @cacheable(strategy='create',
               pathfinder=Cache.pathfinder(file_name_position=1, path_keyword='path'))
    def download_weights(self, file_name):
        self.download_model_files(self.WEIGHTS_FILE, file_name)

    @cacheable(strategy='create',
               pathfinder=Cache.pathfinder(file_name_position=1, path_keyword='path'))
    def download_source_mask_params(self, file_name):
        self.download_model_files(self.SOURCE_MASK_FILE, file_name)

    @cacheable(strategy='create',
               pathfinder=Cache.pathfinder(file_name_position=1, path_keyword='path'))
    def download_target_mask_params(self, file_name):
        self.download_model_files(self.TARGET_MASK_FILE, file_name)

    @cacheable(strategy='create',
               pathfinder=Cache.pathfinder(file_name_position=1, path_keyword='path'))
    def download_connection_density(self, file_name):
        self.download_model_files(self.CONNECTION_DENSITY_FILE, file_name)

    @cacheable(strategy='create',
               pathfinder=Cache.pathfinder(file_name_position=1, path_keyword='path'))
    def download_connection_strength(self, file_name):
        self.download_model_files(self.CONNECTION_STRENGTH_FILE, file_name)

    @cacheable(strategy='create',
               pathfinder=Cache.pathfinder(file_name_position=1, path_keyword='path'))
    def download_normalized_connection_density(self, file_name):
        self.download_model_files(self.NORMALIZED_CONNECTION_DENSITY_FILE, file_name)

    @cacheable(strategy='create',
               pathfinder=Cache.pathfinder(file_name_position=1, path_keyword='path'))
    def download_normalized_connection_strength(self, file_name):
        self.download_model_files(self.NORMALIZED_CONNECTION_STRENGTH_FILE, file_name)
