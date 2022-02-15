import pandas as pd
import numpy as np
from scipy import sparse
import torch
from torch.utils.data import Dataset, DataLoader
import os
from rich.progress import track


class SemanticSimilarityDataset(Dataset):

    def __init__(self, data_directory):
        interpro_dataset = pd.read_table(os.path.join(data_directory, 'interpro.tab'))
        ip_features = interpro_dataset.columns[~interpro_dataset.columns.isin(['Protein accession'])].to_numpy()
        self.interpro_dict = {}
        self.ss_dataset = pd.read_table(os.path.join(data_directory, 'bp-ss.tab'))
        for protein in track(interpro_dataset['Protein accession'].unique(), description='Building InterPro dictionary'):
            self.interpro_dict[protein] = interpro_dataset[
                interpro_dataset['Protein accession'] == protein][ip_features].to_numpy().flatten()

    def __len__(self):
        return self.ss_dataset.shape[0]

    def __getitem__(self, item):
        protein1 = self.ss_dataset.iloc[item].protein1
        protein2 = self.ss_dataset.iloc[item].protein2
        similarity = self.ss_dataset.iloc[item].scaled_similarity
        return (torch.from_numpy(self.interpro_dict[protein1].astype(np.float32)),
                torch.from_numpy(self.interpro_dict[protein2].astype(np.float32)),
                torch.from_numpy(np.array([similarity]).astype(np.float32)))

class MultiTaskSemanticSimilarityDataset(Dataset):

    def __init__(self, data_directory, string_columns):
        interpro_dataset = pd.read_table(os.path.join(data_directory, 'interpro.tab'))
        ip_features = interpro_dataset.columns[~interpro_dataset.columns.isin(['Protein accession'])].to_numpy()
        self.string_columns = string_columns
        self.interpro_dict = {}
        ss_dataset = pd.read_table(os.path.join(data_directory, 'bp-ss.tab'))
        string_nets = pd.read_table(os.path.join(data_directory, "string_nets.tab"))
        string_nets = string_nets[["protein1", "protein2"] + self.string_columns]
        self.multitask_dataset = ss_dataset.merge(string_nets, how="left")
        for protein in track(interpro_dataset['Protein accession'].unique(),
                             description='Building InterPro dictionary'):
            self.interpro_dict[protein] = interpro_dataset[
                interpro_dataset['Protein accession'] == protein][ip_features].to_numpy().flatten()

    def __len__(self):
        return self.multitask_dataset.shape[0]

    def __getitem__(self, item):
        protein1 = self.multitask_dataset.iloc[item].protein1
        protein2 = self.multitask_dataset.iloc[item].protein2
        similarity = self.multitask_dataset.iloc[item].scaled_similarity
        ret = [torch.from_numpy(self.interpro_dict[protein1].astype(np.float32)),
               torch.from_numpy(self.interpro_dict[protein2].astype(np.float32)),
               torch.from_numpy(np.array([similarity]).astype(np.float32))]
        for c in self.string_columns:
            value = self.multitask_dataset.iloc[item][c]
            ret.append(torch.from_numpy(np.array([value]).astype(np.float32)))

        return *ret,

class SemanticSimilarityOnDeviceDataset(Dataset):

    def __init__(self, data_directory, device):
        interpro_dataset = pd.read_csv(os.path.join(data_directory, 'interpro.tab'), sep='\t')
        ip_features = interpro_dataset.columns[~interpro_dataset.columns.isin(['Protein accession'])].to_numpy()
        self.interpro_dict = {}
        self.ss_dataset = pd.read_csv(os.path.join(data_directory, 'bp-ss.tab'), sep='\t')
        for protein in track(interpro_dataset['Protein accession'].unique(),
                             description='Building InterPro dictionary'):
            self.interpro_dict[protein] = torch.from_numpy(
                interpro_dataset[interpro_dataset['Protein accession'] == protein][ip_features]
                    .to_numpy().flatten().astype(np.float32)
            ).to(device)
        self.ss_dataset['scaled_similarity'] = self.ss_dataset['scaled_similarity'].apply(
            lambda x:torch.tensor(x, dtype=torch.float32).to(device))

    def __len__(self):
        return self.ss_dataset.shape[0]

    def __getitem__(self, item):
        protein1 = self.ss_dataset.iloc[item].protein1
        protein2 = self.ss_dataset.iloc[item].protein2
        similarity = self.ss_dataset.iloc[item].scaled_similarity
        return self.interpro_dict[protein1], self.interpro_dict[protein2], similarity

class SparseSemanticSimilarityDatasetDevice(Dataset):

    def __init__(self, data_directory, device, sparse_cache=''):
        self.device = device
        self.sparse_cache = sparse_cache
        if self.sparse_cache != '' and os.path.exists(self.sparse_cache):
            print(f'found precomputed tensors, loading from {self.sparse_cache}')
            tensors_dict = torch.load(self.sparse_cache)
            self._P1 = tensors_dict['P1']
            self._P2 = tensors_dict['P2']
            self._y = tensors_dict['y']
        else:
            interpro_dataset = pd.read_csv(os.path.join(data_directory, 'interpro.tab'), sep='\t')
            ip_features = interpro_dataset.columns[~interpro_dataset.columns.isin(['Protein accession'])].to_numpy()
            interpro_dict = {}
            ss_dataset = pd.read_csv(os.path.join(data_directory, 'bp-ss.tab'), sep='\t')
            for protein in track(interpro_dataset['Protein accession'].unique(),
                                 description='Building InterPro dictionary'):
                interpro_dict[protein] = interpro_dataset[
                    interpro_dataset['Protein accession'] == protein][ip_features].to_numpy().flatten()
            len_features = ip_features.shape[0]
            len_dataset = ss_dataset.shape[0]
            self._P1 = np.zeros((len_dataset,len_features), dtype=np.float32)
            self._P2 = np.zeros((len_dataset,len_features), dtype=np.float32)
            self._y = np.zeros((len_dataset,), dtype=np.float32)
            for i in track(range(len_dataset), description='Building dense dataset'):
                p1 = ss_dataset.iloc[i].protein1
                p2 = ss_dataset.iloc[i].protein2
                similarity = ss_dataset.iloc[i].scaled_similarity
                self._P1[i] = interpro_dict[p1]
                self._P2[i] = interpro_dict[p2]
                self._y[i] = similarity
            print('Building sparse matrices')
            P1_sparse = sparse.coo_matrix(self._P1)
            P2_sparse = sparse.coo_matrix(self._P2)
            print('Building sparse tensors')
            self._P1 = torch.sparse_coo_tensor([P1_sparse.row, P1_sparse.col],
                                               P1_sparse.data,
                                               size=self._P1.shape,
                                               dtype=torch.float32)
            self._P2 = torch.sparse_coo_tensor([P2_sparse.row, P2_sparse.col],
                                               P2_sparse.data,
                                               size=self._P2.shape,
                                               dtype=torch.float32)
            self._y = torch.tensor(self._y, dtype=torch.float32)
            if self.sparse_cache != '':
                print('saving tensors')
                torch.save({'P1':self._P1, 'P2':self._P2, 'y':self._y}, self.sparse_cache)
        print(f'Copying tensors to {self.device}')
        self._P1 = self._P1.to(self.device)
        self._P2 = self._P2.to(self.device)
        self._y = self._y.to(self.device)

    def __len__(self):
        return self._y.shape[0]

    def __getitem__(self, item):
        return self._P1[item], self._P2[item], self._y[item]