import numpy as np
import pandas as pd
from prot2vec.tools.log import setup_logger
from prot2vec.tools.datasets import get_prot2vec_features
from prot2vec.tools.parsers import parse_scop_protein_fasta
from prot2vec.tools.utils import zscore_to_pvalue
from scipy.spatial.distance import pdist, squareform
from rich.progress import track
import logging




def run(representation_file: str, fasta_file: str, seed: int, out_file: str) -> None:
    log = logging.getLogger("prot2vec")
    log.info(f"Loading representation file: {representation_file}")
    dataset = get_prot2vec_features(representation_file)
    dataset["protein"] = dataset["protein"].astype(str)

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

    log.info("Calculating pairwise distances...")
    features = dataset.columns[~dataset.columns.isin(["protein", "set"])]
    distances = squareform(pdist(np.stack(dataset[features].values), metric="cosine"))
    z_score_samples = 1000
    min_proteins = 5 # minimum number of proteins to consider in each group
    vcounts = scop_df[fasta_type].value_counts()
    test_classes = vcounts[vcounts >= min_proteins].index

    log.info(f"Calculating Z-Score for {test_classes.shape[0]} classes")
    rng = np.random.default_rng(seed)
    test_data = {k:[] for k in [f"SCOP-{fasta_type}", "z-score", "p-value", "size"]}
    for i, scop_class in enumerate(test_classes):
        log.info(f"evaluating {scop_class}: ({i/test_classes.shape[0] * 100.0:.2f}%)")
        cond = scop_df[fasta_type] == scop_class
        prots = scop_df[cond]["protein_id"].values
        size = prots.shape[0]
        cond = dataset["protein"].isin(prots)
        idx = dataset[cond].index
        x = distances[idx, :][:, idx][np.triu_indices(idx.shape[0], k=1)].mean()
        background = np.empty(z_score_samples)
        for j in track(range(z_score_samples), total=z_score_samples):
            idx = rng.choice(dataset.shape[0], size=size, replace=False)
            background[j] = distances[idx, :][:, idx][np.triu_indices(idx.shape[0], k=1)].mean()
        m = background.mean()
        s = background.std()
        if s == 0:
            z = 0.0
        else:
            z = (x - m)/s
        test_data[f"SCOP-{fasta_type}"].append(scop_class)
        test_data["z-score"].append(z)
        test_data["p-value"].append(zscore_to_pvalue(z))
        test_data["size"].append(size)
    
    log.info(f"Writing results to {out_file}")
    pd.DataFrame(test_data).to_csv(out_file, index=False, sep="\t")

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
    args = parser.parse_args()
    run(args.representation_file, args.fasta_file, args.seed, args.output_file)

