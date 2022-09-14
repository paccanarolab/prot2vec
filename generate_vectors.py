import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tools.datasets import FastSemanticSimilarityDataset
from models import (SiameseSimilarityNet, SiameseSimilarityPerceptronNet,
                    SiameseSimilaritySmall, SiameseSimilaritySmallPerceptron,
                    SiameseSimilarityMultiTask)
from models import load_checkpoint
from itertools import product
from torch.utils.data import DataLoader
from rich.progress import track
from Utils import Configuration
import os

def extract_proteins_representation_dict(device, model, dataset, set_label, data):
    for prot, numpy_repr in track(dataset.interpro_dict.items(),
                                  description=f'processing {set_label} set...'):
        p = torch.from_numpy(numpy_repr.astype(np.float32)).to(device).reshape(1, -1)
        v = model.prot2vec(p)
        data['protein'].append(prot)
        data['vector'].append(v.flatten().cpu().detach().numpy())
        data['set'].append(set_label)

def extract_proteins_representation(device, model, dataset, set_label, data):
    proteins = dataset.interpro_df.index.values
    n = 50
    for i in range(0, len(proteins), n):
        ps = proteins[i:i + n]
        ip = dataset.interpro_df.loc[proteins[i:i + n]].values
        ip = torch.from_numpy(ip.astype(np.float32)).to(device)
        vs = model.prot2vec(ip)
        for j, p in enumerate(ps):
            data["protein"].append(p)
            data["vector"].append(vs[j].flatten().cpu().detach().numpy())
            data["set"].append(set_label)


def run(run_config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')
    config = Configuration.load_run(run_config)

    dir_train = config["dataset"]["dir_train"]
    dir_val = config["dataset"]["dir_val"]
    dir_test = config["dataset"]["dir_test"]

    batch_size_train = config["model"]["batch_size_train"]
    batch_size_val = config["model"]["batch_size_val"]
    batch_size_test = config["model"]["batch_size_test"]

    interpro_pca = config["dataset"]["interpro_pca"]

    print('Loading training set...')
    ss_bp_train = FastSemanticSimilarityDataset(dir_train,
                                                batch_size=batch_size_train,
                                                shuffle=True, interpro_pca=interpro_pca, 
                                                ignore_pairwise=True)
    print('Loading validation set...')
    ss_bp_val = FastSemanticSimilarityDataset(dir_val,
                                              batch_size=batch_size_val,
                                              shuffle=True, interpro_pca=interpro_pca,
                                              ignore_pairwise=True)
    print('Loading test set...')
    ss_bp_test = FastSemanticSimilarityDataset(dir_test,
                                               batch_size=batch_size_test,
                                               shuffle=True, interpro_pca=interpro_pca, 
                                               ignore_pairwise=True)

    model_classes = [
        SiameseSimilarityNet, SiameseSimilarityPerceptronNet,
        SiameseSimilaritySmall, SiameseSimilaritySmallPerceptron
    ]
    model_classes = [SiameseSimilarityNet]

    activations = ['relu', 'sigmoid']
    activations = ['sigmoid']

    num_interpro_features = ss_bp_test.interpro_df.shape[1]

    num_epochs = config["training"]["num_epochs"]

    negative_sampling = config["dataset"]["negative_sampling"]
    combine_string = config["dataset"]["combine_string"]
    string_columns = config["dataset"]["string_columns"]

    if string_columns[0] == "None":
        cols = None
        string_columns = None
    else:
        cols = ["STRING"] if combine_string else string_columns

    secondary_tasks = []
    if config["dataset"]["include_homology"]:
        secondary_tasks.append("homology")
    if config["dataset"]["include_biogrid"]:
        secondary_tasks.append("BIOGRID")
    if cols:
        secondary_tasks += cols

    for model_class, activation in product(model_classes, activations):

        if model_class == SiameseSimilarityMultiTask:
            model = model_class(activation=activation,
                                dim_first_hidden_layer=config["model"]["dim_first_hidden_layer"],
                                tasks_columns=secondary_tasks).to(device)
        else:
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
        print(f'loading {save_name} with alias {alias}')
        save_name = save_name if alias == "infer" else alias
        save_filename_model = os.path.join(config["model"]["dir_model_output"], save_name)

        load_checkpoint(model, optimizer, save_filename_model)

        model.eval()
        with torch.no_grad():
            data = {'protein':[], 'vector':[], 'set':[]}
            extract_proteins_representation(device, model, ss_bp_test, 'test', data)
            extract_proteins_representation(device, model, ss_bp_train, 'train', data)
            extract_proteins_representation(device, model, ss_bp_val, 'validation', data)
            df = pd.DataFrame(data)

        representation_file = os.path.join(
            config["representations"]["dir_representations_out"],
            f'{save_name}-representations.pkl'
        )
        print(f'saving representation to pickled file: {representation_file}')
        df.to_pickle(representation_file)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Generate vectors from a prot2vec model.")
    parser.add_argument("--run-config", "--c",
                        help="Path to the Configuration file",
                        required=True)
    args = parser.parse_args()
    run(args.run_config)
