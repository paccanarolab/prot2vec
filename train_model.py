import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tools.datasets import SemanticSimilarityDataset
from tools.utils import TrainingProgress
from datetime import datetime
import matplotlib.pyplot as plt
from models import (SiameseSimilarityNet, SiameseSimilarityPerceptronNet,
                    SiameseSimilaritySmall, SiameseSimilaritySmallPerceptron)
from models import count_parameters, save_checkpoint
from itertools import product


def run():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')

    ss_bp_train = SemanticSimilarityDataset('../83333/train_data/')
    ss_bp_val = SemanticSimilarityDataset('../83333/val_data/')
    ss_bp_test = SemanticSimilarityDataset('../83333/test_data/')

    # ss_bp_train = SemanticSimilarityOnDeviceDataset('../83333/train_data/', device)
    # ss_bp_val = SemanticSimilarityOnDeviceDataset('../83333/val_data/', device)
    # ss_bp_test = SemanticSimilarityOnDeviceDataset('../83333/test_data/', device)

    # ss_bp_train = SemanticSimilarityDatasetDevice('E:/prot2vec/83333/train_data/',
    #                                               device, 'E:/prot2vec/83333/train_data/tensor.pt')
    # ss_bp_val = SemanticSimilarityDatasetDevice('E:/prot2vec/83333/val_data/',
    #                                             device, 'E:/prot2vec/83333/val_data/tensor.pt')
    # ss_bp_test = SemanticSimilarityDatasetDevice('E:/prot2vec/83333/test_data/',
    #                                              device, 'E:/prot2vec/83333/test_data/tensor.pt')

    for length, d in zip(['train', 'validation', 'test'], [ss_bp_train, ss_bp_val, ss_bp_test]):
        print(length, len(d))

    train_loader = DataLoader(ss_bp_train, batch_size=256, num_workers=20)
    val_loader = DataLoader(ss_bp_val, batch_size=175, num_workers=20, shuffle=True)

    model_classes = [
        SiameseSimilarityNet, SiameseSimilarityPerceptronNet,
        SiameseSimilaritySmall, SiameseSimilaritySmallPerceptron
    ]
    model_classes = [SiameseSimilarityNet, SiameseSimilaritySmall]

    activations = ['relu', 'sigmoid']
    activations = ['relu']

    for model_class, activation in product(model_classes, activations):

        model = model_class(activation=activation).to(device)
        print(f'running model {model.name()}')

        count_parameters(model)

        optimizer = optim.Adam(model.parameters(), lr=0.0006)
        num_epochs = 200
        criterion = nn.MSELoss()
        save_name = f'[{model.name()}]-{num_epochs}_epochs.pt'

        best_val_loss = float("Inf")
        train_losses = []
        val_losses = []

        with TrainingProgress() as progress:
            epochs = progress.add_task("[green]Epochs", progress_type="epochs", total=num_epochs)
            for epoch in range(num_epochs):
                running_loss = 0.0
                model.train()
                training = progress.add_task(f"[magenta]Training [{epoch+1}]",
                                             total=len(train_loader), progress_type="training")
                validation = progress.add_task(f"[cyan]Validation [{epoch+1}]",
                                               total=len(val_loader), progress_type="validation")

                for p1, p2, sim in train_loader:
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
                avg_train_loss = running_loss / len(train_loader)
                train_losses.append(avg_train_loss)

                val_running_loss = 0.0
                with torch.no_grad():
                    model.eval()
                    for p1, p2, sim in val_loader:
                        p1 = p1.to(device)
                        p2 = p2.to(device)
                        sim = sim.to(device)
                        outputs = model(p1, p2)
                        loss = criterion(outputs, sim)
                        val_running_loss += loss.item()
                        progress.advance(validation)
                avg_val_loss = val_running_loss / len(val_loader)
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
        plt.savefig(
            f'E:/prot2vec/loss-evol-[{model.name()}]-[{datetime.today().strftime("%Y-%m-%d-%H-%M-%S")}].png')
        plt.close('all')

    print("Finished Training")


if __name__ == '__main__':
    run()
