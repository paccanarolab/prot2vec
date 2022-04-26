import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tools.datasets import FastMultitaskSemanticSimilarityDataset
from tools.utils import TrainingProgress
from datetime import datetime
import matplotlib.pyplot as plt
from models import SiameseSimilarityMultiTask
from models import count_parameters, save_checkpoint
from multitask_losses.weighted_losses import WeightedMSEs
from itertools import product
import pickle
from Utils import Configuration
import os

def run():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')
    config = Configuration.load_run("run-test.ini")

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

    batch_size_train = config["model"]["batch_size_train"]
    batch_size_val = config["model"]["batch_size_val"]
    batch_size_test = config["model"]["batch_size_test"]

    dir_train = config["dataset"]["dir_train"]
    dir_val = config["dataset"]["dir_val"]
    dir_test = config["dataset"]["dir_test"]

    print('Loading training set...')
    ss_bp_train = FastMultitaskSemanticSimilarityDataset(dir_train,
                                                         string_columns,
                                                         batch_size=batch_size_train,
                                                         shuffle=True,
                                                         include_homology=config["dataset"]["include_homology"],
                                                         negative_sampling=negative_sampling,
                                                         combine_string_columns=combine_string)
    print('Loading validation set...')
    ss_bp_val = FastMultitaskSemanticSimilarityDataset(dir_val,
                                                       string_columns,
                                                       batch_size=batch_size_val,
                                                       shuffle=True,
                                                       negative_sampling=negative_sampling,
                                                       combine_string_columns=combine_string)
    print('Loading test set...')
    ss_bp_test = FastMultitaskSemanticSimilarityDataset(dir_test,
                                                        string_columns,
                                                        batch_size=batch_size_test,
                                                        shuffle=True,
                                                        negative_sampling=negative_sampling,
                                                        combine_string_columns=combine_string)

    for length, d in zip(['train', 'validation', 'test'], [ss_bp_train, ss_bp_val, ss_bp_test]):
        print(length, len(d))

    # train_loader = DataLoader(ss_bp_train, batch_size=256, num_workers=20)
    # val_loader = DataLoader(ss_bp_val, batch_size=175, num_workers=20, shuffle=True)

    model_classes = [SiameseSimilarityMultiTask]

    # TODO: add these to the configuration file
    activations = ['relu', 'sigmoid']
    activations = ['sigmoid']


    for model_class, activation in product(model_classes, activations):

        model = model_class(activation=activation,
                            dim_first_hidden_layer=config["model"]["dim_first_hidden_layer"],
                            tasks_columns=secondary_tasks).to(device)

        count_parameters(model)

        optimizer = optim.Adam(model.parameters(),
                               lr=config["optimizer"]["learning_rate"])
        num_epochs = config["training"]["num_epochs"]
        # criterion = nn.MSELoss()
        # tasks_criterion = nn.MSELoss()
        criterion = WeightedMSEs(n_tasks=len(secondary_tasks) + 1,
                                 main_task_factor=config["loss"]["main_task_factor"])
        save_name = f'[{model.name()}]-{num_epochs}_epochs.pt'
        alias = config["model"]["alias"]
        print(f'running {save_name} with alias {alias}')
        save_name = save_name if alias == "infer" else alias
        save_filename_model = os.path.join(config["model"]["dir_model_output"], save_name)

        best_val_loss = float("Inf")
        train_losses = []
        val_losses = []
        train_data = {k:[] for k in ["task", "epoch", "batch", "loss", "grad", "bias_grad"]}
        val_data = {k: [] for k in ["task", "epoch", "batch", "loss"]}
        val_tasks_losses = {k:[] for k in ['sim'] + secondary_tasks}
        indicator_size = len(secondary_tasks) + 1 if negative_sampling else 0

        with TrainingProgress() as progress:
            epochs = progress.add_task("[green]Epochs", progress_type="epochs", total=num_epochs)
            for epoch in range(num_epochs):
                running_loss = 0.0
                task_running_losses = {k:0.0 for k in ["sim"] + secondary_tasks}
                model.train()
                training = progress.add_task(f"[magenta]Training [{epoch+1}]",
                                             total=len(ss_bp_train), progress_type="training")
                validation = progress.add_task(f"[cyan]Validation [{epoch+1}]",
                                               total=len(ss_bp_val), progress_type="validation")

                for batch_num, (p1, p2, *tasks) in enumerate(ss_bp_train):
                    # forward
                    p1 = p1.to(device)
                    p2 = p2.to(device)
                    # sim = sim.to(device)
                    if negative_sampling:
                        indicators = [i.to(device) for i in tasks[-indicator_size:]]
                        tasks = [t.to(device) for t in tasks[:-indicator_size]]
                    else:
                        tasks = [t.to(device) for t in tasks]
                        indicators = None
                    # sim_out, *outputs = model(p1, p2)
                    # sim_loss = criterion(sim_out, sim)
                    # tasks_losses = [tasks_criterion(o, t).to(device) for o, t in zip(outputs, tasks)]
                    outputs = model(p1, p2)
                    for o in outputs:
                        o.retain_grad()
                    loss, individual_losses = criterion(outputs, tasks, indicators)
                    loss.retain_grad()
                    for l in individual_losses:
                        l.retain_grad()
                    # backward and optimize
                    optimizer.zero_grad()

                    loss.backward()
                    if config["training"]["output_debug"]:
                        for i, k in enumerate(["sim"] + secondary_tasks):
                            train_data["task"].append(k)
                            train_data["epoch"].append(epoch)
                            train_data["batch"].append(batch_num)
                            train_data["loss"].append(individual_losses[i].item())
                            if i > 0:
                                if not torch.isnan(individual_losses[i]):
                                    train_data["grad"].append(
                                        model.tasks_heads[k][0]._parameters["weight"].grad.detach().cpu().numpy())
                                    train_data["bias_grad"].append(
                                        model.tasks_heads[k][0]._parameters["bias"].grad.detach().cpu().numpy())
                                else:
                                    train_data["grad"].append(np.nan)
                                    train_data["bias_grad"].append(np.nan)
                            else:
                                train_data["grad"].append(outputs[0].grad.detach().cpu().numpy())
                                train_data["bias_grad"].append(0)

                    optimizer.step()
                    running_loss += loss.item()
                    progress.advance(training)
                n = len(ss_bp_train)
                avg_train_loss = running_loss / n
                train_losses.append(avg_train_loss)

                val_running_loss = 0.0
                val_running_losses = {k: 0.0 for k in ["sim"] + secondary_tasks}
                with torch.no_grad():
                    model.eval()
                    for batch_num, (p1, p2, *tasks) in enumerate(ss_bp_val):
                        p1 = p1.to(device)
                        p2 = p2.to(device)
                        if negative_sampling:
                            indicators = [i.to(device) for i in tasks[-indicator_size:]]
                            tasks = [t.to(device) for t in tasks[:-indicator_size]]
                        else:
                            tasks = [t.to(device) for t in tasks]
                            indicators = None
                        outputs = model(p1, p2)
                        loss, individual_losses = criterion(outputs, tasks, indicators)
                        val_running_loss += loss.item()
                        if config["training"]["output_debug"]:
                            for i, k in enumerate(["sim"] + secondary_tasks):
                                val_data["task"].append(k)
                                val_data["epoch"].append(epoch)
                                val_data["batch"].append(batch_num)
                                val_data["loss"].append(individual_losses[i].item())
                        for i, k in enumerate(["sim"] + secondary_tasks):
                            val_running_losses[k] += individual_losses[i].item()
                        progress.advance(validation)
                n = len(ss_bp_val)
                avg_val_loss = val_running_loss / n
                for k in ["sim"] + secondary_tasks:
                    val_tasks_losses[k].append(val_running_losses[k] / n)
                val_losses.append(avg_val_loss)

                print(f'Epoch [{epoch + 1}/{num_epochs}],Train Loss: {avg_train_loss:.4f},',
                      f'Valid Loss: {avg_val_loss:.4f}',
                      f"task_losses: {', '.join(f'{k[:4]}={l[-1]:.4f}' for k, l in val_tasks_losses.items())}")
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    save_checkpoint(save_filename_model, model, optimizer, best_val_loss)

                progress.tasks[training].visible = False
                progress.tasks[validation].visible = False
                progress.advance(epochs)

        # # plotting of training and validation loss
        # fix, axs = plt.subplots(nrows=4, figsize=(10, 10), facecolor='white')
        # for i, ax in enumerate(axs[:2]):
        #     ax.set_xlabel('Epoch')
        #     ax.set_ylabel('Loss')
        #     ax.grid(ls=':', zorder=1)
        #     if i != 0:
        #         ax.set_yscale('log')
        #         ax.set_ylabel('Loss (log scale)')
        #     ax.plot(train_losses, label='Train Loss')
        #     ax.plot(val_losses, label="Validation Loss")
        #     ax.legend(loc='upper right')
        # for i, ax in enumerate(axs[2:]):
        #     ax.set_xlabel('Epoch')
        #     ax.set_ylabel('Loss')
        #     ax.grid(ls=':', zorder=1)
        #     if i != 0:
        #         ax.set_yscale('log')
        #         ax.set_ylabel('Loss (log scale)')
        #     for k, losses in train_tasks_losses.items():
        #         ax.plot(losses, label=f"{k} (train)")
        #     for k, losses in val_tasks_losses.items():
        #         ax.plot(losses, label=f"{k} (validation)")
        #     ax.legend(loc='upper right')
        #
        basename = os.path.join(
            config["training"]["dir_training_out"],
            f"loss-evol-[{save_name}]-[{datetime.today().strftime('%Y-%m-%d-%H-%M-%S')}]"
        )

        # TODO: delete this for production
        basename = os.path.join(
            config["training"]["dir_training_out"],
            f"loss-evol"
        )

        figname = f'{basename}.png'
        dataname = f'{basename}.pkl'
        # plt.savefig(figname)
        # plt.close('all')
        if config["training"]["output_debug"]:
            with open(dataname, 'wb') as f:
                pickle.dump(train_data, f)
                pickle.dump(val_data, f)
    print("Finished Training")


if __name__ == '__main__':
    run()
