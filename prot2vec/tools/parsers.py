import numpy as np
import pandas as pd
from typing import Dict, Tuple, Union
from scipy import sparse


def extract_uniprot_accession(protein_id: str) -> str:
    """
    Transform a UniProt protein id from the format
    db|accession|ID to simply the accession

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


def parse_scop_protein_fasta(protein_line: str) -> Dict:
    protein, class_id, class_pdbid, class_uniid = (protein_line[1:]
                                                   .strip().split())
    class_key, class_id = class_id.split("=")
    class_pdbid_key, class_pdbid = class_pdbid.split("=")
    class_uniid_key, class_uniid = class_uniid.split("=")
    return {
        "protein_id": protein,
        class_key: class_id,
        class_pdbid_key: class_pdbid,
        class_uniid_key: class_uniid
    }


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

    # (GO annotations (e.g. GO:0005515) - optional column;
    # only displayed if –goterms option is switched on)
    INTERPRO_GOTERM_COLUMN = "GO term"
    # (Pathways annotations (e.g. REACT_71) - optional column;
    # only displayed if –pathways option is switched on)"""
    INTERPRO_PATHWAY_COLUMN = "Pathway"

    def __init__(self,
                 interpro_file: str,
                 goterm: bool = False,
                 pathway: bool = False,
                 clean_accessions: bool = True) -> None:
        self.infile = interpro_file
        self.columns = InterProParser.INTERPRO_COLUMNS.copy()
        self.clean_accessions = clean_accessions
        if goterm:
            self.columns.append(InterProParser.INTERPRO_GOTERM_COLUMN)
        if pathway:
            self.columns.append(InterProParser.INTERPRO_PATHWAY_COLUMN)

    def parse(self, ret_type: str = 'numpy') -> Union[pd.DataFrame, Tuple]:
        if ret_type == 'pandas':
            return self._parse_to_pandas()
        elif ret_type == 'dataset':
            return self._parse_to_dataset()
        return self._parse_to_numpy()

    def _parse_to_pandas(self) -> pd.DataFrame:
        df = pd.read_csv(self.infile, sep='\t', names=self.columns)
        if self.clean_accessions:
            df['Protein accession'] = df['Protein accession'].apply(
                extract_uniprot_accession)
        return df

    def _parse_to_dataset(self) -> pd.DataFrame:
        df = self._parse_to_pandas()
        data = df[['Protein accession', 'InterPro accession']].copy()
        condition = data["InterPro accession"] != "-"
        data = data[condition].drop_duplicates().reset_index(drop=True)
        data['value'] = 1
        protein_index = (
            pd.DataFrame(
                enumerate(np.sort(data["Protein accession"].unique())),
                columns=["protein idx", "Protein accession"])
            .set_index("Protein accession"))
        interpro_index = (
            pd.DataFrame(
                enumerate(np.sort(data["InterPro accession"].unique())),
                columns=["interpro idx", "InterPro accession"])
            .set_index("InterPro accession"))
        data = (data.merge(protein_index,
                           left_on="Protein accession",
                           right_index=True)
                .merge(interpro_index,
                       left_on="InterPro accession",
                       right_index=True))
        M = sparse.coo_matrix(
            (
                data["value"],
                (data["protein idx"].values, data["interpro idx"].values)
            )
        )
        return pd.DataFrame(
                data=M.toarray(),
                index=protein_index.index,
                columns=interpro_index.index).reset_index()

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
            a dictionary containing the semantic similarity
            for every pair of proteins
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
            a pandas.DataFrame containing the semantic similarity
            for every pair of proteins

        """
        d = {'protein1': [], 'protein2': [], 'similarity': []}
        proteins = []
        header_read = False
        for line in open(self.filename):
            if line[0] == '!':  # it is a comment line
                continue
            if not header_read:
                header_read = True
                proteins = line.strip().split('\t')
                continue
            parts = line.strip().split('\t')
            if parts[0] not in proteins:
                proteins.append(parts[0])
            for i, sim in enumerate([float(a) for a in parts[1:]]):
                d['protein1'].append(parts[0])
                d['protein2'].append(proteins[i])
                d['similarity'].append(sim)
        return np.array(proteins), pd.DataFrame(d)


def parse_biogrid(filename) -> Tuple:
    def get_uniprot_accession(biogrid_alt_id):
        alt_ids = biogrid_alt_id.split("|")
        for alt_id in alt_ids:
            if "uniprot" in alt_id:
                return alt_id.split(":")[-1]
        return None

    def clean_score(score_str):
        if score_str == "-":
            return np.nan
        return float(score_str.split(":")[-1])

    df = pd.read_table(filename)
    df["protein1"] = df["Alt IDs Interactor A"].apply(get_uniprot_accession)
    df["protein2"] = df["Alt IDs Interactor B"].apply(get_uniprot_accession)
    df["weight"] = df["Confidence Values"].apply(clean_score)
    df["weight"] = df["weight"].fillna(df["weight"].max())
    df = df[["protein1", "protein2", "weight"]].dropna()
    return np.unique(df[["protein1", "protein2"]].to_numpy().flatten()), df
