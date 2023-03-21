import numpy as np
import pandas as pd
from prot2vec.Utils import Configuration
from prot2vec.tools.log import setup_logger
from prot2vec.tools.datasets import build_dataset, get_interpro_features
from prot2vec.tools.utils import save_list_to_file
from prot2vec.models import GOTermLogisticRegression
from pathlib import Path
import logging

def run(run_config):
    log = logging.getLogger("prot2vec")

    log.info(f"Loading configuration file: {run_config}")
    config = Configuration.load_pfp_lr(run_config) 
    dir_lr_out = Path(config["predictions"]["dir_lr_out"]) / "InterPro"
    dir_lr_out.mkdir(parents=True, exist_ok=True)
    function_assignment_bp = Path(config["training"]["function_assignment_bp"])
    function_assignment_mf = Path(config["training"]["function_assignment_mf"])
    function_assignment_cc = Path(config["training"]["function_assignment_cc"])
    min_protein_count_per_term = config["training"]["min_protein_count_per_term"]
    if min_protein_count_per_term <= 0:
        log.warn(f"min_protein_count_per_term set to {min_protein_count_per_term},"
                 "values smaller than 0 are not valid, setting to 5")
        min_protein_count_per_term = 5

    train_features = Path(config["dataset"]["dir_train"]) / config["dataset"]["interpro_filename"]
    val_features = Path(config["dataset"]["dir_val"]) / config["dataset"]["interpro_filename"]
    test_features = Path(config["dataset"]["dir_test"]) / config["dataset"]["interpro_filename"]
    interpro_pca = config["dataset"]["interpro_pca"]
    num_pca = config["dataset"]["num_pca"]
    features_df = get_interpro_features(train_features,
                                        val_features,
                                        test_features,
                                        fmt="pkl" if interpro_pca else "tsv",
                                        num_features=num_pca)
    for function_assignment, domain in zip([function_assignment_bp, function_assignment_mf, function_assignment_cc],
                                           ["BP", "MF", "CC"]):
        log.info(f"Processing domain {domain}")
        function_assignment = Path(function_assignment)
        lr_out = dir_lr_out / domain
        lr_out.mkdir(exist_ok=True)
        X, y, features, protein_index, terms_index, set_index = build_dataset(features_df, 
                                                                              function_assignment)
        lr = GOTermLogisticRegression(lr_out)
        sample_mask = np.isin(set_index, ["train", "validation"])
        log.info(f"Filtering GO terms with < {min_protein_count_per_term}"
                 f" and >= {sample_mask.sum()} annotations")
        annotation_count = y[sample_mask, :].sum(axis=0)
        label_mask = ((annotation_count >= min_protein_count_per_term) &
                      # to remove cases where the GO term is annotated to every protein.
                      (annotation_count < sample_mask.sum())) 

        log.info("Saving features, proteins and terms") 
        features_save = dir_lr_out / f"{domain}-features.txt"
        proteins_save = dir_lr_out / f"{domain}-proteins.txt"
        terms_save = dir_lr_out / f"{domain}-terms.txt"
        save_list_to_file(features, features_save)
        save_list_to_file(protein_index[sample_mask], proteins_save)
        save_list_to_file(terms_index[label_mask], terms_save)
        
        log.info(f"Training started") 
        lr.fit(X[sample_mask, :],
               y[sample_mask][:, label_mask],
               terms_index=terms_index[label_mask])
    log.info("Done")
    
if __name__ == "__main__":
    import argparse

    setup_logger("prot2vec")
    parser = argparse.ArgumentParser("Train a PFP model based on protein representations using Logistic Regression")
    parser.add_argument("--run-config", "--c",
                        help="Path to the run configuration file",
                        required=True)
    args = parser.parse_args()
    run(args.run_config)
   

