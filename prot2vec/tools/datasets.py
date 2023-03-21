import pandas as pd
import numpy as np
from scipy import sparse
import torch
from torch.utils.data import Dataset, DataLoader
import os
import logging
from rich.progress import track
from abc import abstractmethod
from better_abc import ABCMeta, abstract_attribute
from typing import Union, List, Tuple
from pathlib import Path

log = logging.getLogger(__name__)

class SemanticSimilarityDataset(Dataset):

    def __init__(self, data_directory):
        interpro_dataset = pd.read_table(os.path.join(data_directory, 'interpro.tab'))
        ip_features = interpro_dataset.columns[~interpro_dataset.columns.isin(['Protein accession'])].to_numpy()
        self.interpro_dict = {}
        self.ss_dataset = pd.read_table(os.path.join(data_directory, 'semantic-similarity.tab'))
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


def load_dataset(data_directory,
                 string_columns: Union[List[str], None] = None,
                 include_homology=True,
                 include_biogrid=True,
                 negative_sampling=False,
                 combine_string_columns=True, 
                 interpro_pca=False,
                 num_pca = -1,
                 ignore_pairwise=False, 
                 interpro_filename="interpro.tab", 
                 semantic_similarity_filename="semantic-similarity.tab") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads and serves a dataset from a directory containing compatible files:
    - interpro.tab
    - homology.tab
    - semantic-similarity.tab
    - string_nets.tab

    Depending on the arguments passed to this function, files other than `interpro.tab` and
    `semantic-similarity.tab` will become optional

    Parameters
    ----------
    data_directory : Path
        The directory that must contain all files to be loaded
    string_columns : list of str, default None
        The columns of string_nets.tab that will be used. If None, the STRING databse will not be loaded
    include_homology : bool, default True
        whether to include homology as a task
    include_biogrid : bool, default True
        whether to include BIOGRID as a task
    negative_sampling : bool, optional, default `False`
        This applies only to string columns. if `True`, a random negative sampling will be used to balance each
        of the columns indicated by `string_columns`.
    combine_string_columns : bool, default False
        This applies only to string columns. If `True`
    interpro_pca : bool, default False
        If `True`, load a pickled file instead of the usual table, the pickled file contains the PCA representaiton
        of the InterPro features
    num_pca : int, default -1
        Only relevant when `intrerpro_pca` is `True`. This number can be used to limit the number of principal 
        components that will be kept for training. If the number if higher than the number of features in the 
        `interpro_filename`, the program will fail.
    ignore_pairwise : bool, default False
        If True, the semantic similarity file will not be read and an empty pandas DataFrame will be returned instead.
        Only useful when using the dataset wrapper to generate vectors.
    interpro_filename : Path, default "interpro.tab"
        The name of the interpro file that will be loaded.
    semantic_similarity_filename : Path, default "semantic-similarity.tab"
        The name of the semantic similarity file that will be loaded.
    """
    interpro_dict = {}

    if interpro_pca:
        interpro_dataset = pd.read_pickle(os.path.join(data_directory, interpro_filename))
        if num_pca != -1:
            ip_features = interpro_dataset.columns[~interpro_dataset.columns.isin(['Protein accession'])].to_numpy()
            interpro_dataset = interpro_dataset[["Protein accession"] + ip_features[:num_pca].tolist()]
    else:
        interpro_dataset = pd.read_table(os.path.join(data_directory, interpro_filename))
    if include_homology:
        homology_dataset = pd.read_table(os.path.join(data_directory, 'homology.tab'))
        homology_dataset.columns = ["protein1", "protein2", "homology"]
    if include_biogrid:
        biogrid_dataset = pd.read_table(os.path.join(data_directory, 'biogrid.tab'))
        biogrid_dataset.columns = ["protein1", "protein2", "BIOGRID"]
    include_string = string_columns is not None
    ip_features = interpro_dataset.columns[~interpro_dataset.columns.isin(['Protein accession'])].to_numpy()
    if ignore_pairwise:
        pairwise_dataset = pd.DataFrame()
    else:
        ss_dataset = pd.read_table(os.path.join(data_directory, semantic_similarity_filename), names=["protein1", "protein2", "similarity"])
        if include_string:
            string_nets = pd.read_table(os.path.join(data_directory, "string_nets.tab"))
            string_nets = string_nets[["protein1", "protein2"] + string_columns]

        pairwise_dataset = ss_dataset
        if include_string:
            pairwise_dataset = pairwise_dataset.merge(string_nets, how="left")
        if include_homology:
            pairwise_dataset = pairwise_dataset.merge(homology_dataset, how="left")
        if include_biogrid:
            pairwise_dataset = pairwise_dataset.merge(biogrid_dataset, how="left")

        if include_string and combine_string_columns:
            pairwise_dataset["STRING"] = pairwise_dataset[string_columns].mean(axis=1)
            keep = [c for c in pairwise_dataset if c not in string_columns]
            pairwise_dataset = pairwise_dataset[keep]

        for c in pairwise_dataset.columns:
            if c not in ["protein1", "protein2"]:
                m, M = pairwise_dataset[c].min(), pairwise_dataset[c].max()
                R = M - m
                pairwise_dataset[c] = ((pairwise_dataset[c] - m) / R).fillna(0)

    for protein in track(interpro_dataset['Protein accession'].unique(), description='Building InterPro dictionary'):
        interpro_dict[protein] = interpro_dataset[
            interpro_dataset['Protein accession'] == protein][ip_features].to_numpy().flatten()

    interpro_df = pd.DataFrame(interpro_dict).T

    if negative_sampling and not ignore_pairwise:
        # TODO: remove the seed
        rng = np.random.default_rng(0)
        #                                                         these columns don't require negative sampling
        cols = [c for c in pairwise_dataset.columns if c not in ["protein1",
                                                                  "protein2",
                                                                  "similarity",
                                                                  "homology"]]
        # add other columns that will be put under negative sampling
        for c in cols:
            x_unknowns = pairwise_dataset[pairwise_dataset[c] == 0].index.values
            x_positives = pairwise_dataset[pairwise_dataset[c] != 0].index.values
            num_positives = x_positives.shape[0]
            rng.shuffle(x_unknowns)
            x_negs = x_unknowns[:num_positives]
            indicator_col = f"ind_{c}"
            pairwise_dataset[indicator_col] = False
            pairwise_dataset.loc[x_positives, indicator_col] = True
            pairwise_dataset.loc[x_negs, indicator_col] = True
        if include_homology:
            pairwise_dataset["ind_homology"] = True
        pairwise_dataset["ind_similarity"] = True

    return (interpro_df, pairwise_dataset)


# The Fast Datasets where implemented following this dicussion
class FastDataset(metaclass=ABCMeta):
    """
    Faster batched dataset for PyTorch based on the following dicussion:
    https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/5
    """

    dataset_len : int = abstract_attribute()
    batch_size : int = abstract_attribute()
    n_batches: int = abstract_attribute()
    shuffle : bool = abstract_attribute()

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

    @abstractmethod
    def get_tensors_(self, indices):
        raise NotImplementedError()


class FastSemanticSimilarityDataset(FastDataset):
    """
    Loads and serves a dataset from a directory containing compatible files:
    - interpro.tab
    - semantic-similarity.tab

    Parameters
    ----------
    data_directory : Path
        The directory that must contain all files to be loaded
    batch_size : int, default 32
        Batch size
    shuffle : bool, default False
        whether to shuffle the data on each epoch
    """
    def __init__(self,
                 data_directory,
                 batch_size=32,
                 shuffle=False,
                 interpro_pca=False, 
                 num_pca=-1,
                 ignore_pairwise=False, 
                 interpro_filename="interpro.tab",
                 semantic_similarity_filename="semantic-similarity.tab"):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.interpro_df, self.pairwise_dataset = load_dataset(
            data_directory, string_columns=None,
            include_biogrid=False, include_homology=False,
            negative_sampling=False, combine_string_columns=False,
            interpro_pca=interpro_pca, num_pca=num_pca, ignore_pairwise=ignore_pairwise, 
            interpro_filename=interpro_filename, semantic_similarity_filename=semantic_similarity_filename
        )
        self.dataset_len = self.pairwise_dataset.shape[0]
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def get_tensors_(self, indices):
        batch_data = self.pairwise_dataset.iloc[indices]
        return (torch.from_numpy(self.interpro_df.loc[batch_data.protein1].values.astype(np.float32)),
                torch.from_numpy(self.interpro_df.loc[batch_data.protein2].values.astype(np.float32)),
                torch.from_numpy(batch_data["similarity"].values[:, np.newaxis].astype(np.float32)))


class FastMultitaskSemanticSimilarityDataset(FastDataset):
    """
    Loads and serves a dataset from a directory containing compatible files:
    - interpro.tab
    - homology.tab
    - semantic-similarity.tab
    - string_nets.tab

    Parameters
    ----------
    data_directory : Path
        The directory that must contain all files to be loaded
    string_columns : list of str or None
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
    def __init__(self,
                 data_directory,
                 string_columns: Union[List[str], None],
                 batch_size=32,
                 shuffle = False,
                 include_homology=True,
                 include_biogrid=True,
                 negative_sampling = False,
                 combine_string_columns = False):
        self.string_columns = string_columns
        self.include_string = string_columns is not None
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.include_homology = include_homology
        self.include_biogrid = include_biogrid
        self.negative_sampling = negative_sampling
        self.combine_string_columns = combine_string_columns
        self.interpro_df, self.multitask_dataset = load_dataset(
            data_directory,
            string_columns=self.string_columns,
            include_homology=self.include_homology,
            include_biogrid=self.include_biogrid,
            negative_sampling=self.negative_sampling,
            combine_string_columns=self.combine_string_columns
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


class MultiTaskSemanticSimilarityDataset(Dataset):

    def __init__(self, data_directory, string_columns, negative_sampling = False, combine_string_columns = True):
        """
        Loads and serves a dataset from a directory containing compatible files:
        - interpro.tab
        - homology.tab
        - semantic-similarity.tab
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
        self.interpro_dict, self.multitask_dataset = load_dataset(data_directory,
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
        self.ss_dataset = pd.read_csv(os.path.join(data_directory, 'semantic-similarity.tab'), sep='\t')
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
            ss_dataset = pd.read_csv(os.path.join(data_directory, 'semantic-similarity.tab'), sep='\t')
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

def build_dataset(features_df:pd.DataFrame,
                  function_assignment_filename:Union[str, Path],
                  goterms: Union[str, List[str]]="all") -> Tuple:
    
    """
    Builds a scikit-learn style dataset from pre-processed files

    Parameters
    ----------
    features_df: pd.DataFrame
        Features to use per protein
    function_assignment_filename : Path
        File with GO terms assigned to proteins, this should be a two column file in 
        TSV format.
    goterms : str or list of str, default "all"
        A list of GO terms to consider for building the dataset, if "all", then all
        goterns in `function_assignment` will be considered.
    fmt : str, default "pkl"
        The format of the `features_filename` file. Can be "pkl" or "tsv"

    Returns
    -------
    X : array (n_proteins, n_features)
        features
    y : array (n_proteins, n_terms)
        labels (empyty if `function_assignment` is set to None)
        the encoding is binary, and each column is suited to train a Logistic Regression
    features : list of str
        Features included in `X`
    protein_index : list 
        protein index, comaptible with both `X` and `y`
    terms_index : list
        GO term index, compatible with the labels matrix
    set_index : list
        indicates to which set each sample of `X` and `y` belong, values are
        "train", "validation", "test", where "train" were used to learn the
        protein representations.
    """
    features = features_df.columns[~features_df.columns.isin(["protein", "set"])].to_numpy()
    log.info(f"Loading function assignment file: {function_assignment_filename}")
    annotations = pd.read_table(function_assignment_filename)
    annotations.columns = ["protein", "goterm"]
    if goterms != "all":
        annotations = annotations[annotations["goterm"].isin(goterms)]
    annotations = annotations.merge(features_df[["protein", "set"]])

    log.info("Building dataset")
    y = annotations[["protein", "goterm"]].drop_duplicates()
    y["value"] = 1
    y = y.pivot("protein", "goterm", "value").fillna(0).reset_index()

    terms_index = y.columns[~y.columns.isin(["protein"])].to_numpy()
    dataset = features_df.merge(y)
    protein_index = dataset["protein"].to_numpy()
    set_index = dataset["set"].to_numpy()

    return (dataset[features].values, dataset[terms_index].values, 
            features, protein_index, terms_index, set_index)

def get_prot2vec_features(features_filename:Union[str, Path]) -> pd.DataFrame:
    """
    Builds a DataFrame dataset from pre-processed files

    Parameters
    ----------
    features_filename : Path
        Features to use per protein, this should be a pickled file generated with the
        generate_vectors.py script.

    Returns
    -------
    pd.DataFrame
        features for all proteins
    """
    log.info(f"Loading protein representations file: {features_filename}")
    vecs = pd.read_pickle(features_filename)
    num_features = vecs.iloc[0].vector.shape[0]
    features = [f"prot2vec_{i}" for i in range(num_features)]
    vecs = pd.concat([vecs, pd.DataFrame(vecs["vector"].to_list(),
                                         index=vecs.index,
                                         columns=features)], axis=1)
    vecs = vecs[["protein", "set"] + features]
    return vecs


def get_interpro_features(train_file:Union[str, Path],
                          val_file:Union[str, Path], 
                          test_file:Union[str, Path], 
                          fmt="tsv", 
                          num_features:int=-1) -> pd.DataFrame:
    log.info("Loading Interpro Files")
    interpro_dataset = None
    for filename, set_label in zip([train_file, val_file, test_file], ["train", "validation", "test"]):
        log.info(f"Loading {set_label} file: {filename}")
        if fmt == "pkl":
            df = pd.read_pickle(filename)
        elif fmt == "tsv":
            df = pd.read_table(filename)
        df["set"] = set_label
        if interpro_dataset is None:
            interpro_dataset = df
        else:
            interpro_dataset = pd.concat([interpro_dataset, df])
    ip_features = interpro_dataset.columns[~interpro_dataset.columns.isin(["Protein accession", "set"])].to_numpy()
    if num_features != -1:
        log.info(f"Reducing the number of InterPro features to {num_features}")
        ip_features = ip_features[:num_features]
    interpro_dataset = interpro_dataset[["Protein accession", "set"] + ip_features.tolist()]
    interpro_dataset.columns = ["protein", "set"] + ip_features.tolist() 
    return interpro_dataset
