import numpy as np
import pandas as pd
from prot2vec.tools.log import setup_logger
from prot2vec.tools.datasets import get_prot2vec_features
from prot2vec.tools.parsers import parse_scop_protein_fasta
from sklearn.cluster import KMeans
from rich.progress import track
import logging
import sys

def run(representation_file: str, fasta_file: str, out_file: str, 
        mode: str = "prot2vec", 
        min_proteins: int=5,
        n_clusters: int=-1) -> None:
    log = logging.getLogger("prot2vec")
    log.info(f"Loading representation file: {representation_file} with mode {mode}")
    accession_col = "protein" if mode == "prot2vec" else "Protein accession"
    if mode == "prot2vec":
        dataset = get_prot2vec_features(representation_file)
    else:
        dataset = pd.read_table(representation_file)

    log.info(f"Loading SCOP classes from fasta file {fasta_file}...")
    fasta_type = "UNK"
    scop_df = [] 
    with open(fasta_file) as fasta:
        for line in fasta:
            if line[0] == ">":
                scop_df.append(parse_scop_protein_fasta(line))
                if fasta_type == "UNK":
                    fasta_type = "SF" if "SF" in scop_df[0].keys() else "FA"
                    log.info(f"Fasta file contains SCOP-{fasta_type} classes")
    scop_df = pd.DataFrame(scop_df)

    log.info("Extracting features...")
    not_feature_cols = ["protein", "set"] if mode == "prot2vec" else ["Protein accession"]
    features = dataset.columns[~dataset.columns.isin(not_feature_cols)]
    min_proteins = min_proteins # minimum number of proteins to consider in each group
    vcounts = scop_df[fasta_type].value_counts()
    test_classes = vcounts[vcounts >= min_proteins].index
    cond = scop_df[fasta_type].isin(test_classes)
    proteins = scop_df[cond]["protein_id"].unique()
    dataset[accession_col] = dataset[accession_col].astype(str)
    log.info(f"Filtering dataset with {proteins.shape[0]} valid proteins...")
    cond = dataset[accession_col].isin(proteins)
    dataset = dataset[cond]
    if n_clusters > test_classes.shape[0]:
        log.info("stopping because configured k is larger than the true number of classes")
        sys.exit(1)
    k = test_classes.shape[0] if n_clusters == -1 else n_clusters
    log.info(f"Training K-means with k={k}...")
    k_means = KMeans(n_clusters=k)
    X_transformed = k_means.fit_transform(dataset[features].values) 
    dataset["cluster"] = np.argmin(X_transformed, axis=1)
    log.info(f"Writing results to {out_file}")
    dataset[[accession_col, "cluster"]].to_csv(out_file, index=False, sep="\t")
    log.info("Done")

if __name__ == '__main__':
    import argparse

    setup_logger("prot2vec")
    parser = argparse.ArgumentParser("test SCOP categories within a prot2vec space")
    parser.add_argument("--representation-file",
                        help="Path to the protein representations file",
                        required=True)
    parser.add_argument("--fasta-file",
                        help="Path to the fasta file",
                        required=False)
    parser.add_argument("--output-file",
                        help="Path to the output file",
                        required=False)
    parser.add_argument("--min-proteins",
                        help="Minimum number of proteins to consider a class for prediction",
                        type=int,
                        default=5)
    parser.add_argument("--num-clusters",
                        help="number of clusters to get "
                             "-1 indicates the same as the number of classes.",
                        type=int,
                        default=-1)
    parser.add_argument("--mode",
                        help="type of feature to be found in the representations file",
                        default="prot2vec", choices=["prot2vec", "interpro"])
    args = parser.parse_args()
    run(args.representation_file, args.fasta_file, args.output_file, 
        mode=args.mode, min_proteins=args.min_proteins, n_clusters=args.num_clusters)

