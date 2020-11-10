from torchvision.models import resnet18
from torch import nn
import torch
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class SimCLR(nn.Module):

    def __init__(self, proj_dim, temperature):
        super(SimCLR, self).__init__()
        self.encoder = nn.ModuleList(list(resnet18(pretrained=True, progress=True).children())[:-1])
        self.temperature = temperature
        self.proj_head = nn.Sequential(
            nn.Linear(512, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )
        self.criterion = nn.CrossEntropyLoss()
        self.cos = nn.CosineSimilarity(dim=2)

    def negative_mask(self, N):
        mask = np.eye(2 * N)
        mask = mask + np.eye(2 * N, 2 * N, -N) + np.eye(2 * N, 2 * N, N)
        mask = 1 - mask
        mask = torch.from_numpy(mask).type(torch.bool)
        return mask.to(device)

    def forward(self, xi, xj):
        hi, hj = xi, xj
        for layer in self.encoder:
            hi, hj = layer(hi), layer(hj)
        hi, hj = hi.squeeze(), hj.squeeze()
        zi, zj = self.proj_head(hi), self.proj_head(hj)

        s = self.pairwise_similarity(zi, zj)
        Loss = self.get_loss(s)

        return Loss

    def pairwise_similarity(self, zi, zj):
        # zi, zj: [N, D]
        concat = torch.cat([zi, zj], dim=0)  # positive pair: (i, N+i)
        row = concat.unsqueeze(1)  # [2N, 1, D]
        col = concat.unsqueeze(0)  # [1, 2N, D]
        s = self.cos(row, col) / self.temperature  # [2N, 2N]

        return s

    def get_loss(self, s):
        N = int(s.size(0) / 2)

        s_ij = torch.diag(s, N)  # [N, 1]
        s_ji = torch.diag(s, -N)  # [N, 1]
        positive_pair = torch.cat([s_ij, s_ji], dim=0).reshape(2 * N, 1)

        negative_pair = s[self.negative_mask(N)].reshape(2 * N, -1)

        labels = torch.zeros(2 * N).to(device).long()
        logits = torch.cat([positive_pair, negative_pair], dim=1)

        loss = self.criterion(logits, labels)
        return loss

    def get_representation(self, x):
        h = x
        for layer in self.encoder:
            h = layer(h)

        # return self.proj_head(h.squeeze())
        return h.squeeze()


class LinearRegression(nn.Module):

    def __init__(self, in_features, out_features):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)