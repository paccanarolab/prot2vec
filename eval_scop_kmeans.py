import numpy as np
import pandas as pd
from prot2vec.tools.log import setup_logger
from prot2vec.tools.datasets import get_prot2vec_features
from prot2vec.tools.parsers import parse_scop_protein_fasta
from sklearn.cluster import KMeans
from rich.progress import track
import logging




def run(representation_file: str, fasta_file: str, seed: int, out_file: str, mode: str = "prot2vec", n_jobs: int=-1) -> None:
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
    min_proteins = 5 # minimum number of proteins to consider in each group
    vcounts = scop_df[fasta_type].value_counts()
    test_classes = vcounts[vcounts >= min_proteins].index
    log.info("Training K-means...")
    k_means = KMeans(n_clusters=test_classes.shape[0])
    X_transformed = k_means.fit_transform(dataset[features].values) 
    dataset["cluster"] = np.argmin(X_transformed)
    
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
    parser.add_argument("--seed",
                        help="seed for the random number generator",
                        type=int,
                        default=0)
    parser.add_argument("--num-cpu",
                        help="number of CPU cores to use while calcualting "
                             "pairwise distances. -1 indicates ALL cores.",
                        type=int,
                        default=-1)
    parser.add_argument("--mode",
                        help="type of feature to be found in the representations file",
                        default="prot2vec", choices=["prot2vec", "interpro"])
    args = parser.parse_args()
    run(args.representation_file, args.fasta_file, args.seed, args.output_file, 
        mode=args.mode, n_jobs=args.num_cpu)

