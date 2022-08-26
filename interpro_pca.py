from sklearn.decomposition import IncrementalPCA, PCA
from tools.parsers import InterProParser
from pathlib import Path
import numpy as np
import pickle as pk
import logging

def run(logger, n_components):

    pca_file = Path(f"../interpro.pca-{n_components}.pkl")
    X_projected_path = Path(f"../interpro-pca-{n_components}-projected.npy")

    logger.info("Loading InterPro file")
    ip = InterProParser("../goa_all_annotated_uniprot.fasta.interpro")
    ip_dataset = ip.parse(ret_type="dataset")
    ip_proteins = ip_dataset["Protein accession"].unique()
    ip_features = ip_dataset.columns[~ip_dataset.columns.isin(["Protein accession"])].to_numpy()

    X = ip_dataset[ip_features].values

    if not pca_file.exists():
        logger.info("Calculating PCA")
        pca = IncrementalPCA(n_components=n_components)
        pca = PCA(n_components=n_components)
        pca.fit(X)

        logger.info("Saving PCA pickle")
        pk.dump(pca, pca_file.open("wb"))
    else:
        logger.info("Found pre-calculated PCA file, loading")
        pca = pk.load(pca_file.open("rb"))

    logger.info("Transforming data..")
    X_projected = pca.transform(X)
    np.save(X_projected_path, X_projected)


if __name__ == "__main__":
    logger = logging.getLogger("InterPro PCA")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    run(logger, 8800)
