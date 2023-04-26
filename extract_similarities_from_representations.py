import pandas as pd
from sklearn.metrics import pairwise_distances
from prot2vec.tools.datasets import get_prot2vec_features
from prot2vec.tools.log import setup_logger
from scipy.special import comb
from itertools import combinations
from rich.progress import track
import logging

def run(representation_file:str, output_file: str, 
        mode: str = "prot2vec", n_jobs: int = -1) -> None:
    log = logging.getLogger("prot2vec")
    log.info(f"Loading representation file: {representation_file} with mode {mode}")
    accession_col = "protein" if mode == "prot2vec" else "Protein accession"
    distance_metric = "cosine" if mode == "prot2vec" else "jaccard"
    not_feature_cols = ["protein", "set"] if mode == "prot2vec" else ["Protein accession"]
    if mode == "prot2vec":
        dataset = get_prot2vec_features(representation_file)
    else:
        dataset = pd.read_table(representation_file)
    log.info(f"shape: {dataset.shape}")
    features = dataset.columns[~dataset.columns.isin(not_feature_cols)]
    log.info("calculating similarities...")
    similarities = pairwise_distances(dataset[features].values, metric=distance_metric, n_jobs=n_jobs)
    similarities = 1 - similarities
    log.info("Writing similarity file...")
    protein_index = {}
    for idx_accession, accession in dataset[accession_col].to_dict().items():
        protein_index[accession] = idx_accession
    with open(output_file, "w") as distout:
        total = int(comb(len(protein_index), 2, exact=True))
        for p1, p2 in track(combinations(protein_index.keys(), 2), 
                            description="Writing...", 
                            total=total):
            p1i = protein_index[p1]
            p2i = protein_index[p2]
            sim = similarities[p1i, p2i]
            distout.write(f"{p1}\t{p2}\t{sim}\n")
    log.info("Done")
if __name__ == '__main__':
    import argparse

    setup_logger("prot2vec")
    parser = argparse.ArgumentParser("Extract cosine or jaccard similariteis between representations")
    parser.add_argument("--representation-file",
                        help="Path to the protein representations file",
                        required=True)
    parser.add_argument("--output-file",
                        help="Path to the output file",
                        required=False)
    parser.add_argument("--num-cpu",
                        help="number of CPU cores to use while calcualting "
                             "pairwise distances. -1 indicates ALL cores.",
                        type=int,
                        default=-1)
    parser.add_argument("--mode",
                        help="type of feature to be found in the representations file",
                        default="prot2vec", choices=["prot2vec", "interpro"])
    args = parser.parse_args()
    run(args.representation_file, args.output_file, 
        mode=args.mode, n_jobs=args.num_cpu)

