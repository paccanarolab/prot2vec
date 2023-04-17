import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from prot2vec.tools.datasets import FastSemanticSimilarityDataset
from prot2vec.models import (SiameseSimilarityNet, SiameseSimilarityPerceptronNet,
                            SiameseSimilaritySmall, SiameseSimilaritySmallPerceptron,
                            SiameseSimilarityMultiTask)
from prot2vec.models import load_checkpoint
from itertools import product
from rich.progress import track
from prot2vec.Utils import Configuration
from prot2vec.tools.log import setup_logger
from pathlib import Path
import os
import sys
import logging


log = logging.getLogger("prot2vec")

def extract_proteins_representation_dict(device, model, dataset, set_label, data):
    for prot, numpy_repr in track(dataset.interpro_dict.items(),
                                  description=f'processing {set_label} set...'):
        p = torch.from_numpy(numpy_repr.astype(np.float32)).to(device).reshape(1, -1)
        v = model.prot2vec(p)
        data['protein'].append(prot)
        data['vector'].append(v.flatten().cpu().detach().numpy())
        data['set'].append(set_label)

def extract_proteins_representation(device, model, dataset, set_label, data):
    proteins = dataset.index.values
    n = 50
    for i in range(0, len(proteins), n):
        ps = proteins[i:i + n]
        ip = dataset.loc[proteins[i:i + n]].values
        ip = torch.from_numpy(ip.astype(np.float32)).to(device)
        vs = model.prot2vec(ip)
        for j, p in enumerate(ps):
            data["protein"].append(p)
            data["vector"].append(vs[j].flatten().cpu().detach().numpy())
            data["set"].append(set_label)


def run(run_config, interpro_file, output_file):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log.info(f'Using {device} device')
    config = Configuration.load_run(run_config)

    interpro_pca = config["dataset"]["interpro_pca"]
    if interpro_pca:
        sys.exit(1)

    log.info('Creating dataset from InterPro file...')
    dataset = pd.read_table(interpro_file)

    log.info('Loading validation set to synchronize feature set..')
    dir_val = config["dataset"]["dir_val"]
    interpro_filename = config["dataset"]["interpro_filename"]
    ss_bp_val = pd.read_table(Path(f"{dir_val}/{interpro_filename}"), nrows=1)

    log.info("Setting dataset features to the compatible set...")
    dataset = dataset.reindex(columns=ss_bp_val.columns, fill_value=0)
    del ss_bp_val

    interpro_dict = {}
    ip_features = dataset.columns[~dataset.columns.isin(['Protein accession'])].to_numpy()
    for protein in track(dataset['Protein accession'].unique(), description='Building InterPro dictionary'):
        interpro_dict[protein] = dataset[
            dataset['Protein accession'] == protein][ip_features].to_numpy().flatten()

    dataset = pd.DataFrame(interpro_dict).T
    del interpro_dict

    model_classes = [
        SiameseSimilarityNet, SiameseSimilarityPerceptronNet,
        SiameseSimilaritySmall, SiameseSimilaritySmallPerceptron
    ]
    model_classes = [SiameseSimilarityNet]

    activations = ['relu', 'sigmoid']
    activations = ['sigmoid']

    num_interpro_features = dataset.shape[1]

    num_epochs = config["training"]["num_epochs"]

    for model_class, activation in product(model_classes, activations):
        log.info("Configuring model...")
        model = model_class(num_interpro_features,
                            activation=activation,
                            dim_first_hidden_layer=config["model"]["dim_first_hidden_layer"]
                            ).to(device)
        # this must be the same as the one used in train_model.py
        # for the time being, this is hardcoded
        optimizer = optim.Adam(model.parameters(),
                               lr=config["optimizer"]["learning_rate"])

        save_name = f'[{model.name()}]-{num_epochs}_epochs.pt'
        alias = config["model"]["alias"]
        log.info(f'loading {save_name} with alias {alias}...')
        save_name = save_name if alias == "infer" else alias
        save_filename_model = os.path.join(config["model"]["dir_model_output"], save_name)

        load_checkpoint(model, optimizer, save_filename_model)

        model.eval()
        log.info("Calculating vectors...")
        with torch.no_grad():
            data = {'protein':[], 'vector':[], 'set':[]}
            extract_proteins_representation(device, model, dataset, 'test', data)
            df = pd.DataFrame(data)

        log.info(f'saving representation to pickled file: {output_file}...')
        df.to_pickle(output_file)
    log.info('Done')


if __name__ == '__main__':
    import argparse

    setup_logger("prot2vec")
    parser = argparse.ArgumentParser(description="Generate vectors using a prot2vec model and "
                                                 "a parsed InterPro file.")
    parser.add_argument("--run-config", "--c",
                        help="Path to the Configuration file used to train the model",
                        required=True)
    parser.add_argument("--interpro-file", "--i",
                        help="Path to the parsed interpro file in tab separated format",
                        required=True)
    parser.add_argument("--output-file", "--o",
                        help="Path to the output file",
                        required=True)
    args = parser.parse_args()
    run(args.run_config, args.interpro_file, args.output_file)
