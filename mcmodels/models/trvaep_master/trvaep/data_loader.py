import numpy as np
import torch
from scipy import sparse
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

from trvaep.utils import remove_sparsity


def label_encoder(adata, label_encoder=None, condition_key='condition'):
    if label_encoder is None:
        le = LabelEncoder()
        labels = le.fit_transform(adata.obs[condition_key].tolist())
    else:
        le = label_encoder
        labels = np.zeros(adata.shape[0])
        for condition, label in label_encoder.items():
            labels[adata.obs[condition_key] == condition] = label
    return labels.reshape(-1, 1), le


class CustomDatasetFromAdata(Dataset):
    def __init__(self, adata, condition_key=None):
        self.condtion_key = condition_key
        self.adata = adata
        if sparse.issparse(self.adata.X):
            self.adata = remove_sparsity(self.adata)
        self.data = np.array(self.adata.X)
        if self.condtion_key is not None:
            self.labels, self.le = label_encoder(self.adata, condition_key=condition_key)
            self.labels = np.array(self.labels)

    def __getitem__(self, index):
        if self.condtion_key is not None:
            single_cell_label = self.labels[index]
            label_as_tensor = torch.Tensor(single_cell_label)
        single_cell_expression = self.data[index, :]
        cell_as_tensor = torch.Tensor(single_cell_expression)
        if self.condtion_key is not None:
            return cell_as_tensor, label_as_tensor
        else:
            return cell_as_tensor, None

    def __len__(self):
        return len(self.adata)

    def get_label_ecnoder(self):
        return self.le
