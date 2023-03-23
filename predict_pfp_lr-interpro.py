import numpy as np
import pandas as pd
from prot2vec.Utils import Configuration
from prot2vec.tools.log import setup_logger
from prot2vec.tools.datasets import get_interpro_features
from prot2vec.models import GOTermLogisticRegression
from pathlib import Path
from rich.progress import track
import logging


def run(run_config):
    log = logging.getLogger("prot2vec")
    log.info(f"Loading configuration file: {run_config}")
    config = Configuration.load_pfp_lr(run_config) 
    dir_lr_out = Path(config["predictions"]["dir_lr_out"]) / "InterPro"
    pred_out = config["predictions"]["predictions_out_file"]
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
        outfile = Path(f"{pred_out}-{domain}-IP.tsv")
        if outfile.exists():
            log.info(f"{outfile} exists, skipping computation")
            continue
        function_assignment = Path(function_assignment)
        lr_out = dir_lr_out / domain
        lr = GOTermLogisticRegression(lr_out)
        lr.load_trained_model()
        condition = features_df["set"] == "test"
        features = features_df.columns[~features_df.columns.isin(["protein", "set"])].to_numpy()
        X_test = features_df[condition][features].to_numpy()
        tot_terms = len(lr.models_)
        predictions = np.empty((X_test.shape[0], tot_terms))
        for i, (_, model) in track(enumerate(lr.models_.items()), 
                                      description="Predicting...",
                                      total=tot_terms):
            predictions[:, i] = model.predict_proba(X_test)[:, 1]
        log.info("Saving prediction file")
        t_index = list(lr.models_.keys())
        p_df = pd.DataFrame(data=predictions,
                            columns=t_index)
        p_df["protein"] = features_df[condition]["protein"]
        p_df[["protein"] + t_index].to_csv(outfile, sep="\t", index=False)
        
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
   

