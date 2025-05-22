import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import utils


class GATLayer(nn.Module):
    def __init__(self, dimension, sampleSize, DKG, layer):
        super().__init__()
        self.DKG = DKG.long()
        self.sampleSize = sampleSize
        self.drugNumber = len(torch.unique(DKG[:, 0]))
        self.relationNumber = len(torch.unique(DKG[:, 2]))
        self.tailNumber = len(torch.unique(DKG[:, 1]))
        self.dimension = dimension
        self.layer = layer
        self.drugEmbeding = nn.Embedding(num_embeddings=self.drugNumber, embedding_dim=dimension)
        self.relationEmbeding = nn.Embedding(num_embeddings=self.relationNumber, embedding_dim=dimension)
        self.tailEmbeding = nn.Embedding(num_embeddings=self.tailNumber, embedding_dim=dimension)
        self.fullConnectionLayer = nn.Sequential(
            nn.Linear(dimension, dimension)
        )
        self.fullConnectionLayer2 = nn.Sequential(
            nn.Linear(dimension * 2, dimension),
            nn.BatchNorm1d(dimension)
        )
        self.fullConnectionLayer3 = nn.Sequential(
            nn.Linear(dimension, dimension, bias=False)
        )
        self.fullConnectionLayer4 = nn.Sequential(
            nn.Linear(dimension, dimension, bias=False)
        )
        self.fullConnectionLayer5 = nn.Sequential(
            nn.Linear(dimension * 2, 1, bias=False)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                if (m.bias != None):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)
                else:
                    gain = nn.init.calculate_gain('relu')
                    nn.init.xavier_normal_(m.weight, gain=gain)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, arguments):
        if self.layer == 0:
            X = arguments
        elif self.layer == 1:
            embedding1, X = arguments
        elif self.layer == 2:
            embedding1, embedding2, X = arguments
        elif self.layer == 3:
            embedding1, embedding2, embedding3, X = arguments

        updim_Embeding1 = self.fullConnectionLayer3(self.drugEmbeding(self.DKG[:, 0]))
        updim_Embeding2 = self.fullConnectionLayer4(self.tailEmbeding(self.DKG[:, 1]))

        drug_tail = torch.cat([updim_Embeding1, updim_Embeding2], dim=1)
        att = F.leaky_relu(self.fullConnectionLayer5(drug_tail))

        hadamardProduct = self.drugEmbeding(self.DKG[:, 0]) * self.relationEmbeding(self.DKG[:, 2])

        semanticsFeatureScore = torch.sum(self.fullConnectionLayer(hadamardProduct), dim=1).reshape((-1, 1))

        tempEmbedding = semanticsFeatureScore * self.tailEmbeding(self.DKG[:, 1])

        neighborhoodEmbedding = torch.zeros(self.drugNumber, self.dimension).to('cuda:1')

        for i in range(self.drugNumber):

            length = torch.sum(self.DKG[:, 0] == i)
            if length == 0:
                continue
            if length >= self.sampleSize:

                index = list(utils.data.WeightedRandomSampler(self.DKG[:, 0] == i, self.sampleSize, replacement=False))

                alpha = F.softmax(att[index], dim=0)

                neighborhoodEmbedding[i] = torch.sum(tempEmbedding[index] * alpha, dim=0)

            else:
                index = (self.DKG[:, 0] == i).nonzero().squeeze().tolist()

                alpha = F.softmax(att[index], dim=0)

                neighborhoodEmbedding[i] = torch.sum(tempEmbedding[index] * alpha, dim=0)

        concatenate = torch.cat([self.drugEmbeding.weight, neighborhoodEmbedding], 1)

        if self.layer == 0:
            return self.fullConnectionLayer2(concatenate), X
        elif self.layer == 1:
            return embedding1, self.fullConnectionLayer2(concatenate), X
        elif self.layer == 2:
            return embedding1, embedding2, self.fullConnectionLayer2(concatenate), X
        elif self.layer == 3:
            return embedding1, embedding2, embedding3, self.fullConnectionLayer2(concatenate), X
