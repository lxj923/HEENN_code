import torch
import torch.nn as nn


# self, dimension, sampleSize, DKG, pLayers, layer, ddi=False
class EncoderLayer(nn.Module):
    def __init__(self, dimension, DKG, id):
        super(EncoderLayer, self).__init__()
        self.DKG = DKG.long()
        self.drugNumber = len(torch.unique(DKG[:, 0]))  # 唯一蛋白质的数量
        self.tailNumber = len(torch.unique(DKG[:, 1]))  # 唯一实体数量
        self.id = id

        self.input_r = torch.zeros((self.drugNumber, self.tailNumber), dtype=torch.float32).to('cuda')
        if id == 0:
            for m in DKG:
                #self.input_r[m[0 ], m[1]] = m[-1] + 1
                self.input_r[m[0], m[1]] = 1
        elif id == 1:
            for m in DKG:
                if m[-1] == 1:
                    self.input_r[m[0], m[1]] = 1
                    self.input_r[m[1], m[0]] = 1

        self.encoder = nn.Sequential(
            nn.Linear(self.tailNumber, int(self.tailNumber / 2)),
            nn.BatchNorm1d(int(self.tailNumber / 2)),
            nn.Softmax(dim=1),
            # nn.ReLU(),
            nn.Linear(int(self.tailNumber / 2), dimension),
            # nn.Linear(self.tailNumber, dimension)
        )

        # 在PyTorch中，self.modules() 函数用于递归地遍历模型的所有模块，包括模型本身和它所有的子模块。
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 对于线性层，使用Xavier均匀初始化来初始化权重
                nn.init.xavier_uniform_(m.weight)
                # 将偏置项初始化为0
                nn.init.zeros_(m.bias)

    def forward(self, arguments):
        if self.id == 0:
            embedding1, embedding2, X = arguments
        elif self.id == 1:
            embedding1, embedding2, embedding3, X = arguments

        encode = self.encoder(self.input_r)

        if self.id == 0:
            return embedding1, embedding2, encode, X
        elif self.id == 1:
            return embedding1, embedding2, embedding3, encode, X
