from __future__ import division
import mock
import pytest
import numpy as np

from allensdk.core.reference_space import ReferenceSpace
from allensdk.core.structure_tree import StructureTree

@pytest.fixture(scope="session")
def mask():
    mask = mock.Mock()
    mask.mask_volume.side_effect = lambda x: x
    mask._mask.return_value = np.ones((10, 10, 10))

    return mask


@pytest.fixture(scope="session")
def tree():
    return [{'id': 1, 'structure_id_path': [1]},
            {'id': 2, 'structure_id_path': [1, 2]},
            {'id': 3, 'structure_id_path': [1, 3]},
            {'id': 4, 'structure_id_path': [1, 2, 4]},
            {'id': 5, 'structure_id_path': [1, 2, 5]},
            {'id': 6, 'structure_id_path': [1, 2, 5, 6]},
            {'id': 7, 'structure_id_path': [1, 7]}]


@pytest.fixture(scope="session")
def annotation():
    # leaves are 6, 4, 3
    # additionally annotate 2, 5 for realism :)
    annotation = np.zeros((10, 10, 10))
    annotation[4:8, 4:8, 4:8] = 2
    annotation[5:7, 5:7, 5:7] = 5
    annotation[5:7, 5:7, 5] = 6
    annotation[7, 7, 7] = 4
    annotation[8:10, 8:10, 8:10] = 3

    return annotation


@pytest.fixture(scope="session")
def mcc(tree, annotation):
    # data
    shape = (10, 10, 10)

    data_mask = np.ones(shape)
    injection_density = np.ones(shape)
    injection_fraction = np.ones(shape)
    projection_density = np.ones(shape)

    # mock
    mcc = mock.Mock(manifest_file="manifest_file")

    mcc.get_data_mask.return_value = (data_mask, )
    mcc.get_injection_density.return_value = (injection_density, )
    mcc.get_injection_fraction.return_value = (injection_fraction, )
    mcc.get_projection_density.return_value = (projection_density, )

    mcc.get_experiments.return_value = [{"id":456}, {"id":12}, {"id":315}]

    # reference space
    rsp = ReferenceSpace(StructureTree(tree), annotation, [10, 10, 10])
    mcc.get_reference_space.return_value = rsp
    mcc.get_structure_tree.return_value = StructureTree(tree)

    return mcc
