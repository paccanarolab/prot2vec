import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def __init__(self, activation=SiameseSimilarity.DEFAULT_ACTIVATION):
        super(SiameseSimilarityNet, self).__init__(activation)

        self.prot2vec = nn.Sequential(
            nn.Linear(7098, 1024),
            nn.BatchNorm1d(1024),
            self.activation(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            self.activation(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            self.activation(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            self.activation(),
        )

    def forward(self, p1, p2):
        p1 = F.normalize(self.prot2vec(p1))
        p2 = F.normalize(self.prot2vec(p2))
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
        p1 = F.normalize(self.prot2vec(p1))
        p2 = F.normalize(self.prot2vec(p2))
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