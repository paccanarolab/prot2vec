import torch
import torch.nn as nn


class WeightedMSEs(nn.Module):

    def __init__(self, n_tasks: int, main_task_factor = 1.0):
        super(WeightedMSEs, self).__init__()
        self.n_tasks = n_tasks
        self.loss_weights = nn.Parameter(torch.ones(self.n_tasks))
        self.task_factors = [main_task_factor] + [1.0] * (n_tasks -1)

    def forward(self, preds, labels, tasks_indicators=None):
        mse = nn.MSELoss()

        losses = []
        loss = None
        for task in range(self.n_tasks):
            if tasks_indicators is not None:
                losses.append(self.task_factors[task] * self.loss_weights[task] * mse(preds[task][tasks_indicators[task]],
                                                            labels[task][tasks_indicators[task]]))
            else:
                losses.append(self.task_factors[task] * self.loss_weights[task] * mse(preds[task], labels[task]))

            if loss is None:
                loss = losses[-1]
            else:
                if not torch.isnan(losses[-1]):
                    loss += losses[-1]
        return loss, losses
