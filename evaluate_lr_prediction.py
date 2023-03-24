import pandas as pd
from prot2vec.tools.log import setup_logger
from prot2vec.Utils import Configuration
from pathlib import Path
from rich.progress import track
import logging


def run(run_config, mode="prot2vec"):
    log = logging.getLogger("prot2vec") 
    log.info(f"Loading configuration file: {run_config}")
    config = Configuration.load_pfp_lr(run_config) 
    dir_lr_out = Path(config["predictions"]["dir_lr_out"])
    if mode == "interpro":
        dir_lr_out = dir_lr_out / "InterPro"
    function_assignment_bp = Path(config["training"]["function_assignment_bp"])
    function_assignment_mf = Path(config["training"]["function_assignment_mf"])
    function_assignment_cc = Path(config["training"]["function_assignment_cc"])
    for function_assignment, eval_domain in zip([function_assignment_bp, function_assignment_mf, function_assignment_cc],
                                                ["BP", "MF", "CC"]):
        for pred_domain in ["BP", "MF", "CC"]:
            log.info(f"Processing, eval_domain: {eval_domain}, pred_domain: {pred_domain}")
            predictions = pd.read_table(f"{config['predictions']['predictions_out_file']}-{pred_domain}.tsv")
            log.info(f"predictions.shape (initial): {predictions.shape}")
            pred_terms = predictions.columns[~predictions.columns.isin(["protein"])]
            annotations = pd.read_table(function_assignment)
            annotations.columns = ["protein", "goterm"]
            annotations = annotations[annotations["protein"].isin(predictions["protein"])]
            common_terms = sorted(set(pred_terms) & set(annotations["goterm"]))
            annotations = annotations[annotations["goterm"].isin(common_terms)].drop_duplicates()
            annotations["value"] = 1
            annotations = annotations.pivot("protein", "goterm", "value").fillna(0).reset_index()
            predictions = predictions[["protein"] + common_terms]
            log.info(f"predictions.shape: {predictions.shape}")
            log.info(f"annotations.shape: {annotations.shape}")

if __name__ == "__main__":
    import argparse

    setup_logger("prot2vec")
    parser = argparse.ArgumentParser("Train a PFP model based on protein representations using Logistic Regression")
    parser.add_argument("--run-config", "--c",
                        help="Path to the run configuration file",
                        required=True)
    args = parser.parse_args()
    run(args.run_config)
   


