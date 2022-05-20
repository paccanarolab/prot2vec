import torch
import torch.nn as nn
import torch.optim as optim
from tools.datasets import FastSemanticSimilarityDataset
from tools.utils import TrainingProgress
from datetime import datetime
import matplotlib.pyplot as plt
from models import (SiameseSimilarityNet, SiameseSimilarityPerceptronNet,
                    SiameseSimilaritySmall, SiameseSimilaritySmallPerceptron)
from models import count_parameters, save_checkpoint
from itertools import product
from Utils import Configuration
import pickle
import os


def run():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')

    config = Configuration.load_run("run-all_goa.ini")

    batch_size_train = config["model"]["batch_size_train"]
    batch_size_val = config["model"]["batch_size_val"]
    batch_size_test = config["model"]["batch_size_test"]

    dir_train = config["dataset"]["dir_train"]
    dir_val = config["dataset"]["dir_val"]
    dir_test = config["dataset"]["dir_test"]

    print('Loading training set...')
    ss_bp_train = FastSemanticSimilarityDataset(dir_train,
                                                batch_size=batch_size_train,
                                                shuffle=True)
    print('Loading validation set...')
    ss_bp_val = FastSemanticSimilarityDataset(dir_val,
                                              batch_size=batch_size_val,
                                              shuffle=True)
    print('Loading test set...')
    ss_bp_test = FastSemanticSimilarityDataset(dir_test,
                                               batch_size=batch_size_test,
                                               shuffle=True)

    # ss_bp_train = SemanticSimilarityOnDeviceDataset('../83333/train_data/', device)
    # ss_bp_val = SemanticSimilarityOnDeviceDataset('../83333/val_data/', device)
    # ss_bp_test = SemanticSimilarityOnDeviceDataset('../83333/test_data/', device)

    # ss_bp_train = SemanticSimilarityDatasetDevice('E:/prot2vec/83333/train_data/',
    #                                               device, 'E:/prot2vec/83333/train_data/tensor.pt')
    # ss_bp_val = SemanticSimilarityDatasetDevice('E:/prot2vec/83333/val_data/',
    #                                             device, 'E:/prot2vec/83333/val_data/tensor.pt')
    # ss_bp_test = SemanticSimilarityDatasetDevice('E:/prot2vec/83333/test_data/',
    #                                              device, 'E:/prot2vec/83333/test_data/tensor.pt')

    for split_name, d in zip(['train', 'validation', 'test'], [ss_bp_train, ss_bp_val, ss_bp_test]):
        print(split_name, d.dataset_len)
        print(split_name, d.interpro_df.shape)


    model_classes = [
        SiameseSimilarityNet, SiameseSimilarityPerceptronNet,
        SiameseSimilaritySmall, SiameseSimilaritySmallPerceptron
    ]
    model_classes = [SiameseSimilarityNet]

    activations = ['relu', 'sigmoid']
    activations = ['sigmoid']

    num_interpro_features = ss_bp_train.interpro_df.shape[1]

    for model_class, activation in product(model_classes, activations):

        model = model_class(num_interpro_features,
                            activation=activation,
                            dim_first_hidden_layer=config["model"]["dim_first_hidden_layer"]).to(device)

        count_parameters(model)

        optimizer = optim.Adam(model.parameters(),
                               lr=config["optimizer"]["learning_rate"])
        num_epochs = config["training"]["num_epochs"]
        criterion = nn.MSELoss()
        save_name = f'[{model.name()}]-{num_epochs}_epochs.pt'
        alias = config["model"]["alias"]
        print(f'running {save_name} with alias {alias}')
        save_name = save_name if alias == "infer" else alias
        save_filename_model = os.path.join(config["model"]["dir_model_output"], save_name)

        best_val_loss = float("Inf")
        train_losses = []
        val_losses = []

        with TrainingProgress() as progress:
            epochs = progress.add_task("[green]Epochs", progress_type="epochs", total=num_epochs)
            for epoch in range(num_epochs):
                running_loss = 0.0
                model.train()
                training = progress.add_task(f"[magenta]Training [{epoch+1}]",
                                             total=len(ss_bp_train), progress_type="training")
                validation = progress.add_task(f"[cyan]Validation [{epoch+1}]",
                                               total=len(ss_bp_val), progress_type="validation")

                for p1, p2, sim in ss_bp_train:
                    # forward
                    p1 = p1.to(device)
                    p2 = p2.to(device)
                    sim = sim.to(device)
                    outputs = model(p1, p2)
                    loss = criterion(outputs, sim)

                    # backward and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    progress.advance(training)
                n = len(ss_bp_train)
                avg_train_loss = running_loss / n
                train_losses.append(avg_train_loss)

                val_running_loss = 0.0
                with torch.no_grad():
                    model.eval()
                    for p1, p2, sim in ss_bp_val:
                        p1 = p1.to(device)
                        p2 = p2.to(device)
                        sim = sim.to(device)
                        outputs = model(p1, p2)
                        loss = criterion(outputs, sim)
                        val_running_loss += loss.item()
                        progress.advance(validation)
                avg_val_loss = val_running_loss / len(ss_bp_val)
                val_losses.append(avg_val_loss)

                print('Epoch [{}/{}],Train Loss: {:.4f}, Valid Loss: {:.8f}'
                      .format(epoch + 1, num_epochs, avg_train_loss, avg_val_loss))
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    save_checkpoint(save_name, model, optimizer, best_val_loss)

                progress.tasks[training].visible = False
                progress.tasks[validation].visible = False
                progress.advance(epochs)

        # plotting of training and validation loss
        fix, axs = plt.subplots(nrows=2, figsize=(10, 10), facecolor='white')
        for i, ax in enumerate(axs):
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.grid(ls=':', zorder=1)
            if i != 0:
                ax.set_yscale('log')
                ax.set_ylabel('Loss (log scale)')
            ax.plot(train_losses, label='Train Loss')
            ax.plot(val_losses, label="Validation Loss")
            ax.legend(loc='upper right')
        basename = os.path.join(
            config["training"]["dir_training_out"],
            f"loss-evol-[{save_name}]-[{datetime.today().strftime('%Y-%m-%d-%H-%M-%S')}]"
        )
        figname = f'{basename}.png'
        dataname = f'{basename}.pkl'
        plt.savefig(figname)
        plt.close('all')
        with open(dataname, 'wb') as f:
            pickle.dump(train_losses, f)
            pickle.dump(val_losses, f)

    print("Finished Training")


if __name__ == '__main__':
    run()
