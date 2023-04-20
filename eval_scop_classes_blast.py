import numpy as np
import pandas as pd
from scipy.linalg import blas
from prot2vec.tools.log import setup_logger
from prot2vec.tools.parsers import parse_scop_protein_fasta
from prot2vec.tools.utils import zscore_to_pvalue, assert_lexicographical_order
from rich.progress import track
import logging

def evalue_transform(evalue):
    if evalue <= 0: # lower than 0 is impossible, but I want to include everythin in this range
        return 1.0
    if evalue < 11.0:
        return -np.log(evalue / 11.0)
    return 0.0

def run(blast_file: str, fasta_file: str, seed: int, out_file: str) -> None:
    log = logging.getLogger("prot2vec")
    log.info(f"Loading blast file: {blast_file}")
    names = "qaccver saccver pident length mismatch gapopen qstart qend sstart send evalue bitscore qlen".split()
    blast_entries = pd.read_csv(blast_file, sep='\t', names=names)

    log.info("Processing sequence similarity scores...")
    blast_entries["homology_score"] = blast_entries["evalue"].apply(evalue_transform)
    blast_entries["qaccver"] = blast_entries["qaccver"].astype(str)
    blast_entries["saccver"] = blast_entries["saccver"].astype(str)
    blast_proteins = np.unique(blast_entries[["qaccver", "saccver"]].values.flatten())
    blast_entries = blast_entries[["qaccver", "saccver", "homology_score"]].copy()
    max_hom_score = blast_entries["homology_score"].max()
    cond = blast_entries["qaccver"] == blast_entries["saccver"]
    blast_entries.loc[cond, "homology_score"] = max_hom_score
    blast_entries["normalized_homology_score"] = blast_entries["homology_score"].apply(
        lambda x: max(0, x / max_hom_score))
    assert_lexicographical_order(blast_entries, p1="qaccver", p2="saccver") 
    blast_entries = blast_entries.groupby(["qaccver", "saccver"]).aggregate(
        homology_score = pd.NamedAgg("normalized_homology_score", "max")
    ).reset_index()
    cond = blast_entries["homology_score"] > 0
    blast_entries = blast_entries[cond]
    blast_entries["qaccver"] = blast_entries["qaccver"].apply(lambda x: str(int(x)))
    blast_entries["saccver"] = blast_entries["saccver"].apply(lambda x: str(int(x)))

    log.info("Building sequence similarity graph...")
    n = blast_proteins.shape[0]
    blast_graph = np.zeros((n, n))
    idx_dir = {blast_proteins[i]:i for i in range(n)}
    for i, r in track(blast_entries.iterrows(), total=blast_entries.shape[0]):
        p1i = idx_dir[r["qaccver"]]
        p2i = idx_dir[r["saccver"]]
        blast_graph[p1i, p2i] = r["homology_score"]
        blast_graph[p2i, p1i] = r["homology_score"]

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
        idx = np.where(np.isin(blast_proteins, prots))[0]
        x = blast_graph[idx, :][:, idx][np.triu_indices(idx.shape[0], k=1)].mean()
        background = np.empty(z_score_samples)
        for j in track(range(z_score_samples), total=z_score_samples):
            idx = rng.choice(blast_proteins.shape[0], size=size, replace=False)
            background[j] = blast_graph[idx, :][:, idx][np.triu_indices(idx.shape[0], k=1)].mean()
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
    parser.add_argument("--blast-file",
                        help="Path to the BLAST file",
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
    run(args.blast_file, args.fasta_file, args.seed, args.output_file)

