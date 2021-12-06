import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tools.datasets import SemanticSimilarityDataset
from models import (SiameseSimilarityNet, SiameseSimilarityPerceptronNet,
                    SiameseSimilaritySmall, SiameseSimilaritySmallPerceptron)
from models import load_checkpoint
from itertools import product
from torch.utils.data import DataLoader
from rich.progress import track


def extract_proteins_representation(device, model, dataset, set_label, data):
    for prot, numpy_repr in track(dataset.interpro_dict.items(),
                                  description=f'processing {set_label} set...'):
        p = torch.from_numpy(numpy_repr.astype(np.float32)).to(device).reshape(1, -1)
        v = F.normalize(model.prot2vec(p))
        data['protein'].append(prot)
        data['vector'].append(v.flatten().cpu().detach().numpy())
        data['set'].append('train')

def run():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')

    ss_bp_train = SemanticSimilarityDataset('../83333/train_data/')
    ss_bp_val = SemanticSimilarityDataset('../83333/val_data/')
    ss_bp_test = SemanticSimilarityDataset('../83333/test_data/')

    model_classes = [
        SiameseSimilarityNet, SiameseSimilarityPerceptronNet,
        SiameseSimilaritySmall, SiameseSimilaritySmallPerceptron
    ]
    model_classes = [SiameseSimilarityNet, SiameseSimilaritySmall]

    activations = ['relu', 'sigmoid']
    activations = ['relu']
    num_epochs = 200

    for model_class, activation in product(model_classes, activations):

        model = model_class(activation=activation).to(device)

        # this must be the same as the one used in train_model.py
        # for the time being, this is hardcoded
        optimizer = optim.Adam(model.parameters(), lr=0.0006)

        save_name = f'[{model.name()}]-{num_epochs}_epochs.pt'
        load_checkpoint(model, optimizer, save_name)

        model.eval()
        with torch.no_grad():
            data = {'protein':[], 'vector':[], 'set':[]}
            extract_proteins_representation(device, model, ss_bp_train, 'train', data)
            extract_proteins_representation(device, model, ss_bp_val, 'validation', data)
            extract_proteins_representation(device, model, ss_bp_test, 'test', data)
            df = pd.DataFrame(data)

        print('saving representation to pickled file: representations.pkl')
        df.to_pickle('representations.pkl')

if __name__ == '__main__':
    run()