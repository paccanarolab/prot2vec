import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tools.datasets import SemanticSimilarityDataset
from models import (SiameseSimilarityNet, SiameseSimilarityPerceptronNet,
                    SiameseSimilaritySmall, SiameseSimilaritySmallPerceptron,
                    SiameseSimilarityMultiTask)
from models import load_checkpoint
from itertools import product
from torch.utils.data import DataLoader
from rich.progress import track
from Utils import Configuration
import os

def extract_proteins_representation(device, model, dataset, set_label, data):
    for prot, numpy_repr in track(dataset.interpro_dict.items(),
                                  description=f'processing {set_label} set...'):
        p = torch.from_numpy(numpy_repr.astype(np.float32)).to(device).reshape(1, -1)
        v = F.normalize(model.prot2vec(p))
        data['protein'].append(prot)
        data['vector'].append(v.flatten().cpu().detach().numpy())
        data['set'].append(set_label)

def run():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')
    config = Configuration.load_run("run-test.ini")

    dir_train = config["dataset"]["dir_train"]
    dir_val = config["dataset"]["dir_val"]
    dir_test = config["dataset"]["dir_test"]

    print('Loading training set...')
    ss_bp_train = SemanticSimilarityDataset(dir_train)
    print('Loading validation set...')
    ss_bp_val = SemanticSimilarityDataset(dir_val)
    print('Loading test set...')
    ss_bp_test = SemanticSimilarityDataset(dir_test)

    model_classes = [
        SiameseSimilarityNet, SiameseSimilarityPerceptronNet,
        SiameseSimilaritySmall, SiameseSimilaritySmallPerceptron
    ]
    model_classes = [SiameseSimilarityMultiTask]

    activations = ['relu', 'sigmoid']
    activations = ['sigmoid']
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
            model = model_class(activation=activation,
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
            extract_proteins_representation(device, model, ss_bp_train, 'train', data)
            extract_proteins_representation(device, model, ss_bp_val, 'validation', data)
            extract_proteins_representation(device, model, ss_bp_test, 'test', data)
            df = pd.DataFrame(data)

        representation_file = os.path.join(
            config["representations"]["dir_representations_out"],
            f'{save_name}-representations.pkl'
        )
        print(f'saving representation to pickled file: {representation_file}')
        df.to_pickle(representation_file)


if __name__ == '__main__':
    run()