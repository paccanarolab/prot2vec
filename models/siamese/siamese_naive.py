import torch
import torch.nn as nn
import torch.nn.functional as functional


class SiameseSimilarity(nn.Module):

    DEFAULT_ACTIVATION = 'sigmoid'

    def __init__(self, activation=DEFAULT_ACTIVATION):
        super(SiameseSimilarity, self).__init__()
        self.activation_name = activation
        if activation == 'relu':
            self.activation = nn.ReLU
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid

    def name(self):
        return f'{self.__class__.__name__}-{self.activation_name}'


class SiameseSimilarityNet(SiameseSimilarity):

    def __init__(self,
                 num_interpro_features,
                 activation=SiameseSimilarity.DEFAULT_ACTIVATION,
                 dim_first_hidden_layer=1024):
        super(SiameseSimilarityNet, self).__init__(activation)
        self._d1 = dim_first_hidden_layer
        self.prot2vec = nn.Sequential(
            nn.Linear(num_interpro_features, self._d1),
            # nn.BatchNorm1d(self._d1),
            self.activation(),
            nn.Linear(self._d1, self._d1//2),
            # nn.BatchNorm1d(self._d1//2),
            self.activation(),
            nn.Linear(self._d1//2, self._d1//4),
            # nn.BatchNorm1d(self._d1//4),
            self.activation(),
            nn.Linear(self._d1//4, self._d1//4),
            # nn.BatchNorm1d(self._d1//4),
            self.activation(),
        )

    def name(self):
        return f'{super().name()}-{self._d1}'

    def forward(self, p1, p2):
        p1 = functional.normalize(self.prot2vec(p1))
        p2 = functional.normalize(self.prot2vec(p2))
        batch_size = p1.shape[0]
        dim = p1.shape[1]
        return torch.bmm(p1.reshape(batch_size, 1, dim), p2.reshape(batch_size, dim, 1)).squeeze(-1)


class SiameseSimilarityPerceptronNet(SiameseSimilarityNet):

    def __init__(self, activation=SiameseSimilarity.DEFAULT_ACTIVATION):
        super(SiameseSimilarityPerceptronNet, self).__init__(activation)
        self.embedding = nn.Sequential(
            nn.Linear(256 * 2, 256),
            nn.BatchNorm1d(256),
            self.activation(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            self.activation(),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            self.activation(),
            nn.Linear(32, 1),
            self.activation(),
        )

    def forward(self, p1, p2):
        p1 = self.prot2vec(p1)
        p2 = self.prot2vec(p2)
        cat = torch.hstack([p1, p2])
        return self.embedding(cat)


class SiameseSimilaritySmall(SiameseSimilarity):

    def __init__(self, activation=SiameseSimilarity.DEFAULT_ACTIVATION):
        super(SiameseSimilaritySmall, self).__init__(activation)
        self.prot2vec = nn.Sequential(
            nn.Linear(7098, 1024),
            nn.BatchNorm1d(1024),
            self.activation(),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            self.activation(),
        )

    def forward(self, p1, p2):
        p1 = functional.normalize(self.prot2vec(p1))
        p2 = functional.normalize(self.prot2vec(p2))
        batch_size = p1.shape[0]
        dim = p1.shape[1]
        return torch.bmm(p1.reshape(batch_size, 1, dim), p2.reshape(batch_size, dim, 1)).squeeze(-1)


class SiameseSimilaritySmallPerceptron(SiameseSimilaritySmall):

    def __init__(self, activation=SiameseSimilarity.DEFAULT_ACTIVATION):
        super(SiameseSimilaritySmallPerceptron, self).__init__(activation)
        self.embedding = nn.Sequential(
            nn.Linear(256 * 2, 256),
            nn.BatchNorm1d(256),
            self.activation(),
            nn.Linear(256, 32),
            nn.BatchNorm1d(32),
            self.activation(),
            nn.Linear(32, 1),
            self.activation(),
        )

    def forward(self, p1, p2):
        p1 = self.prot2vec(p1)
        p2 = self.prot2vec(p2)
        cat = torch.hstack([p1, p2])
        return self.embedding(cat)


class SiameseSimilarityMultiTask(SiameseSimilarityNet):

    def __init__(self,
                 activation=SiameseSimilarity.DEFAULT_ACTIVATION,
                 dim_first_hidden_layer=1024,
                 tasks_columns=None
                 ):
        super(SiameseSimilarityMultiTask, self).__init__(activation=activation,
                                                         dim_first_hidden_layer=dim_first_hidden_layer)
        self._tasks = tasks_columns if tasks_columns is not None else []
        self._dp2v = self._d1 // 4

        self.tasks_heads = nn.ModuleDict()
        for task in self._tasks:
            self.tasks_heads[task] = nn.Sequential(
                nn.Linear(self._dp2v * 2, self._dp2v),
                self.activation(),
                nn.Linear(self._dp2v, self._dp2v // 2),
                self.activation(),
                nn.Linear(self._dp2v // 2, self._dp2v // 4),
                self.activation(),
                nn.Linear(self._dp2v // 4, self._dp2v // 8),
                self.activation(),
                nn.Linear(self._dp2v // 8, 1),
                self.activation()
            )

    def forward(self, p1, p2):
        self.p1 = functional.normalize(self.prot2vec(p1))
        self.p2 = functional.normalize(self.prot2vec(p2))
        p1_p2 = torch.cat([self.p1, self.p2], 1)
        batch_size = self.p1.shape[0]
        dim = self.p1.shape[1]
        main_head = torch.bmm(self.p1.reshape(batch_size, 1, dim), self.p2.reshape(batch_size, dim, 1)).squeeze(-1)
        secondary_heads = [self.tasks_heads[t](p1_p2) for t in self._tasks]

        return main_head, *secondary_heads
