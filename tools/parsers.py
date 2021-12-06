from rich.progress import track
import numpy as np
import pandas as pd
from typing import Tuple

def extract_uniprot_accession(protein_id: str) -> str:
    """
    Transform a UniProt protein id from the format db|accession|ID to simply the accession

    Parameters
    ----------
    protein_id : str
        UniProt protein id in the format db|accession|ID

    Returns
    -------
    str
        The accession part of the identifier
    """
    return protein_id.split('|')[1]


class InterProParser(object):

    INTERPRO_COLUMNS = ["Protein accession",
        "Sequence MD5 digest",
        "Sequence length",
        "Analysis",
        "Signature accession",
        "Signature description",
        "Start location",
        "Stop location",
        "Score",
        "Status",
        "Date",
        "InterPro accession",
        "InterPro description"]

    # (GO annotations (e.g. GO:0005515) - optional column; only displayed if –goterms option is switched on)
    INTERPRO_GOTERM_COLUMN = "GO term"
    # (Pathways annotations (e.g. REACT_71) - optional column; only displayed if –pathways option is switched on)"""
    INTERPRO_PATHWAY_COLUMN = "Pathway"

    def __init__(self,
                 interpro_file: str,
                 goterm: bool=False,
                 pathway: bool=False) -> None:
        self.infile = interpro_file
        self.columns = InterProParser.INTERPRO_COLUMNS.copy()
        if goterm:
            self.columns.append(InterProParser.INTERPRO_GOTERM_COLUMN)
        if pathway:
            self.columns.append(InterProParser.INTERPRO_PATHWAY_COLUMN)

    def parse(self, ret_type: str = 'numpy'):
        if ret_type == 'pandas':
            return self._parse_to_pandas()
        elif ret_type == 'dataset':
            return self._parse_to_dataset()
        return self._parse_to_numpy()

    def _parse_to_pandas(self) -> pd.DataFrame:
        df = pd.read_csv(self.infile, sep='\t', names=self.columns)
        df['Protein accession'] = df['Protein accession'].apply(extract_uniprot_accession)
        return df

    def _parse_to_dataset(self) -> pd.DataFrame:
        df = self._parse_to_pandas()
        data = df[['Protein accession', 'InterPro accession']].copy()
        data['value'] = 1
        X = (data.drop_duplicates()
                    .pivot('Protein accession', 'InterPro accession', 'value')
                    .fillna(0).reset_index())
        return X[['Protein accession'] + X.columns[~X.columns.isin(['Protein accession'])][1:].tolist()]

    def _parse_to_numpy(self) -> Tuple:
        X = self._parse_to_dataset()
        proteins = X['Protein accession'].unique()
        proteins.sort()
        features = X.columns[~X.columns.isin(['Protein accession'])]
        return proteins, features, X[features].values


class SemanticSimilarityParser(object):

    def __init__(self, filename: str) -> None:
        self.filename = filename

    def load_matrix(self) -> Tuple:
        """
        Loads a similarity matrix file into memory

        Returns
        -------
        proteins : list
            a list of the proteins found in the similarity matrix
        semantic_matrix : dict
            a dictionary containing the semantic similarity for every pair of proteins
        """
        print("Loading Semantic Similarity file: " + self.filename)
        proteins = []
        semantic_matrix = {}
        header_read = False
        for line in open(self.filename):
            if line[0] == '!':  # it is a comment line
                continue
            if not header_read:
                header_read = True
                proteins = line.split('\t')[:-1]
                continue

            parts = line.split('\t')
            if parts[0] not in proteins:
                proteins.append(parts[0])
            semantic_matrix[parts[0]] = [float(a) for a in parts[1:]]

        return np.array(proteins), semantic_matrix

    def load_matrix_pd(self) -> Tuple:
        """
        Similar to `load_matrix`, but returns a pandas.DataFramse instead

        Returns
        -------
        proteins : list
            a list of the proteins found in the similarity matrix
        semantic_matrix : pandas.DataFrame
            a pandas.DataFrame containing the semantic similarity for every pair of proteins

        """
        d = {'protein1': [], 'protein2': [], 'similarity': []}
        proteins = []
        header_read = False
        for line in open(self.filename):
            if line[0] == '!':  # it is a comment line
                continue
            if not header_read:
                header_read = True
                proteins = line.split('\t')[:-1]
                continue
            parts = line.strip().split('\t')
            if parts[0] not in proteins:
                proteins.append(parts[0])
            for i, sim in enumerate([float(a) for a in parts[1:]]):
                d['protein1'].append(parts[0])
                d['protein2'].append(proteins[i])
                d['similarity'].append(sim)
        return np.array(proteins), pd.DataFrame(d)
