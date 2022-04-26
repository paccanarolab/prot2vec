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


def load_multiclass_dataset(data_directory,
                            string_columns,
                            include_homology=True,
                            include_biogrid=True,
                            negative_sampling=False,
                            combine_string_columns=True):
    """
    Loads and serves a dataset from a directory containing compatible files:
    - interpro.tab
    - homology.tab
    - bp-ss.tab
    - string_nets.tab

    Parameters
    ----------
    data_directory : Path
        The directory that must contain all files to be loaded
    string_columns : list of str
        The columns of string_nets.tab that will be used
    include_homology : bool, default True
        whether to include homology as a task
    include_biogrid : bool, default True
        whether to include BIOGRID as a task
    negative_sampling : bool, optional, default `False`
        This applies only to string columns. if `True`, a random negative sampling will be used to balance each
        of the columns indicated by `string_columns`.
    combine_string_columns : bool, default False
        This applies only to string columns. If `True`
    """
    interpro_dict = {}

    interpro_dataset = pd.read_table(os.path.join(data_directory, 'interpro.tab'))
    if include_homology:
        homology_dataset = pd.read_table(os.path.join(data_directory, 'homology.tab'))
        homology_dataset.columns = ["protein1", "protein2", "homology"]
    if include_biogrid:
        biogrid_dataset = pd.read_table(os.path.join(data_directory, 'biogrid.tab'))
        biogrid_dataset.columns = ["protein1", "protein2", "BIOGRID"]
    include_string = string_columns is not None
    ip_features = interpro_dataset.columns[~interpro_dataset.columns.isin(['Protein accession'])].to_numpy()
    ss_dataset = pd.read_table(os.path.join(data_directory, 'bp-ss.tab'))
    if include_string:
        string_nets = pd.read_table(os.path.join(data_directory, "string_nets.tab"))
        string_nets = string_nets[["protein1", "protein2"] + string_columns]

    multitask_dataset = ss_dataset
    if include_string:
        multitask_dataset = multitask_dataset.merge(string_nets, how="left")
    if include_homology:
        multitask_dataset = multitask_dataset.merge(homology_dataset, how="left")
    if include_biogrid:
        multitask_dataset = multitask_dataset.merge(biogrid_dataset, how="left")

    if include_string and combine_string_columns:
        multitask_dataset["STRING"] = multitask_dataset[string_columns].mean(axis=1)
        keep = [c for c in multitask_dataset if c not in string_columns]
        multitask_dataset = multitask_dataset[keep]

    for c in multitask_dataset.columns:
        if c not in ["protein1", "protein2"]:
            m, M = multitask_dataset[c].min(), multitask_dataset[c].max()
            R = M - m
            multitask_dataset[c] = ((multitask_dataset[c] - m) / R).fillna(0)

    for protein in track(interpro_dataset['Protein accession'].unique(), description='Building InterPro dictionary'):
        interpro_dict[protein] = interpro_dataset[
            interpro_dataset['Protein accession'] == protein][ip_features].to_numpy().flatten()

    interpro_df = pd.DataFrame(interpro_dict).T

    if negative_sampling:
        # TODO: remove the seed
        rng = np.random.default_rng(0)
        #                                                         these columns don't require negative sampling
        cols = [c for c in multitask_dataset.columns if c not in ["protein1",
                                                                  "protein2",
                                                                  "similarity",
                                                                  "homology"]]
        # add other columns that will be put under negative sampling
        for c in cols:
            x_unknowns = multitask_dataset[multitask_dataset[c] == 0].index.values
            x_positives = multitask_dataset[multitask_dataset[c] != 0].index.values
            num_positives = x_positives.shape[0]
            rng.shuffle(x_unknowns)
            x_negs = x_unknowns[:num_positives]
            indicator_col = f"ind_{c}"
            multitask_dataset[indicator_col] = False
            multitask_dataset.loc[x_positives, indicator_col] = True
            multitask_dataset.loc[x_negs, indicator_col] = True
        if include_homology:
            multitask_dataset["ind_homology"] = True
        multitask_dataset["ind_similarity"] = True

    return interpro_df, multitask_dataset

# https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/5

class FastMultitaskSemanticSimilarityDataset:
    def __init__(self,
                 data_directory,
                 string_columns,
                 batch_size=32,
                 shuffle = False,
                 include_homology=True,
                 include_biogrid=True,
                 negative_sampling = False,
                 combine_string_columns = False):
        """
        Loads and serves a dataset from a directory containing compatible files:
        - interpro.tab
        - homology.tab
        - bp-ss.tab
        - string_nets.tab

        Parameters
        ----------
        data_directory : Path
            The directory that must contain all files to be loaded
        string_columns : list of str
            The columns of string_nets.tab that will be used
        batch_size : int, default 32
            Batch size
        shuffle : bool, default False
            whether to shuffle the data on each epoch
        include_homology : bool, default True
            whether to include homology as a task
        include_biogrid : bool, default True
            whether to include BIOGRID as a task
        negative_sampling : bool, optional, default `False`
            This applies only to string columns. if `True`, a random negative sampling will be used to balance each
            of the columns indicated by `string_columns`.
        combine_string_columns : bool, default False
            This applies only to string columns. If `True`
        """
        self.string_columns = string_columns
        self.include_string = string_columns is not None
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.include_homology = include_homology
        self.include_biogrid = include_biogrid
        self.negative_sampling = negative_sampling
        self.combine_string_columns = combine_string_columns
        self.interpro_df, self.multitask_dataset = load_multiclass_dataset(
            data_directory, self.string_columns,
            self.include_homology, self.include_biogrid,
            self.negative_sampling, self.combine_string_columns
        )
        self.string_tasks = ["STRING"] if self.combine_string_columns else self.string_columns
        self.dataset_len = self.multitask_dataset.shape[0]
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def get_tensors_(self, indices):
        batch_data = self.multitask_dataset.iloc[indices]
        ret = [torch.from_numpy(self.interpro_df.loc[batch_data.protein1].values.astype(np.float32)),
               torch.from_numpy(self.interpro_df.loc[batch_data.protein2].values.astype(np.float32))]
        cols = ["similarity"]
        if self.include_homology:
            cols.append("homology")
        if self.include_biogrid:
            cols.append("BIOGRID")
        if self.include_string:
            c = ["STRING"] if self.combine_string_columns else self.string_columns
            cols += c
        ind_cols = [f"ind_{c}" for c in cols] if self.negative_sampling else []
        cols += ind_cols
        for c in cols:
            value = batch_data[c].values
            if c.startswith("ind_"):
                ret.append(torch.from_numpy(value[:, np.newaxis].astype(bool)))
            else:
                ret.append(torch.from_numpy(value[:, np.newaxis].astype(np.float32)))
        return *ret,

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        if self.shuffle:
            self.indices = torch.randperm(self.dataset_len)
        else:
            self.indices = None
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        if self.indices is not None:
            indices = self.indices[self.i: self.i+self.batch_size]
            batch = self.get_tensors_(indices)
        else:
            batch = self.get_tensors_(np.arange(self.i, self.i+self.batch_size))
        self.i += self.batch_size
        return batch

class MultiTaskSemanticSimilarityDataset(Dataset):

    def __init__(self, data_directory, string_columns, negative_sampling = False, combine_string_columns = True):
        """
        Loads and serves a dataset from a directory containing compatible files:
        - interpro.tab
        - homology.tab
        - bp-ss.tab
        - string_nets.tab

        Parameters
        ----------
        data_directory : Path
            The directory that must contain all files to be loaded
        string_columns : list of str
            The columns of string_nets.tab that will be used
        negative_sampling : bool, optional, default `False`
            This applies only to string columns. if `True`, a random negative sampling will be used to balance each
            of the columns indicated by `string_columns`.
        combine_string_columns : bool, default False
            This applies only to string columns. If `True`
        """
        self.string_columns = string_columns
        self.negative_sampling = negative_sampling
        self.combine_string_columns = combine_string_columns
        self.interpro_dict, self.multitask_dataset = load_multiclass_dataset(data_directory,
                                                     self.string_columns,
                                                     self.negative_sampling,
                                                     self.combine_string_columns)

    def __len__(self):
        return self.multitask_dataset.shape[0]

    def __getitem__(self, item):
        protein1 = self.multitask_dataset.iloc[item].protein1
        protein2 = self.multitask_dataset.iloc[item].protein2
        similarity = self.multitask_dataset.iloc[item].scaled_similarity
        ret = [torch.from_numpy(self.interpro_dict[protein1].astype(np.float32)),
               torch.from_numpy(self.interpro_dict[protein2].astype(np.float32)),
               torch.from_numpy(np.array([similarity]).astype(np.float32)),]
        cols = ["STRING"] if self.combine_string_columns else self.string_columns
        for c in ["homology"] + cols:
            value = self.multitask_dataset.iloc[item][c]
            ret.append(torch.from_numpy(np.array([value]).astype(np.float32)))

        if self.negative_sampling:
            for c in cols:
                indicator_col = f"ind_{c}"
                value = self.multitask_dataset.iloc[item][indicator_col]
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