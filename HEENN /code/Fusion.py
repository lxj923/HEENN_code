import torch
import torch.nn as nn
import torch.nn.functional as F
import Tool


class FusionLayer(nn.Module):
    # 维度 去掉率 索引
    def __init__(self, dimension, dropout, GNNlayers):
        super().__init__()
        self.GNNlayers = GNNlayers
        self.fullConnectionLayer = nn.Sequential(
            nn.Linear(dimension * 2, dimension),  # 第一个线性层，输入维度是 dimension * 2，输出维度是 dimension
            nn.BatchNorm1d(dimension),  # 对第一个线性层的输出进行批归一化
            nn.Softmax(dim=1),  # 将第一个批归一化的输出转换为概率分布
            nn.Dropout(dropout),  # 应用dropout以减少过拟合
            nn.Linear(dimension, int(dimension / 2)),  # 第二个线性层，输出维度减半
            nn.BatchNorm1d(int(dimension / 2)),  # 对第二个线性层的输出进行批归一化
            nn.Softmax(dim=1),  # 将第二个批归一化的输出转换为概率分布
            # nn.Dropout(dropout),
            nn.Linear(int(dimension / 2), 1),  # 第三个线性层，输出维度为1，通常用于回归或二分类的最后一层
            nn.BatchNorm1d(1),  # 尝试对单个输出进行批归一化
            nn.Sigmoid()  # 将输出压缩到(0, 1)区间，通常用于二分类的输出
        )

        self.fullConnectionLayer2 = nn.Sequential(
            nn.Linear(Tool.net_args.kgDimension, Tool.net_args.ddiDimension)
        )
        self.fullConnectionLayer3 = nn.Sequential(
            nn.Linear(Tool.net_args.kgDimension_enCode, Tool.net_args.ddiDimension)
        )
        self.fullConnectionLayer4 = nn.Sequential(
            nn.Linear(Tool.net_args.ddiDimension_enCode, Tool.net_args.ddiDimension)
        )

        # 初始化权重、偏置
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # if (m.bias != None):
                # 对于线性层，使用Xavier均匀初始化来初始化权重
                nn.init.xavier_uniform_(m.weight)
                # 将偏置项初始化为0
                nn.init.zeros_(m.bias)
            # else:
            #     gain = nn.init.calculate_gain('relu')
            #     nn.init.xavier_normal_(m.weight, gain=gain)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, arguments):
        if self.GNNlayers == 1:
            embedding1, X = arguments
            Embedding = embedding1
        elif self.GNNlayers == 2:
            embedding1, embedding2, X = arguments
            # 将每个蛋白质的两个GNN部分生成的两个特征表示融合在一起，从而得到蛋白质pi的最终嵌入表示
            Embedding = torch.cat([embedding1, embedding2], 1)
        elif self.GNNlayers == 3:
            embedding1, embedding2, embedding3, X = arguments
            Embedding = torch.cat([embedding1, embedding2, embedding3], 1)
        elif self.GNNlayers == 4:
            embedding1, embedding2, embedding3, embedding4, X = arguments

            embedding1_att = self.fullConnectionLayer2(embedding1)
            embedding3_att = self.fullConnectionLayer3(embedding3)
            embedding4_att = self.fullConnectionLayer4(embedding4)


            # 做点积
            att1 = torch.abs(torch.einsum('ij,ij->i', [embedding1_att, embedding2]).reshape(4283, 1))
            att3 = torch.abs(torch.einsum('ij,ij->i', [embedding3_att, embedding2]).reshape(4283, 1))
            att4 = torch.abs(torch.einsum('ij,ij->i', [embedding4_att, embedding2]).reshape(4283, 1))

            att = torch.cat([att1, att3, att4], 1)

            att_score = F.softmax(att, dim=1)

            Embedding = torch.cat([embedding1 * att_score[:, 0].reshape(4283, 1),
                                   embedding2,
                                   embedding3 * att_score[:, 1].reshape(4283, 1),
                                   embedding4 * att_score[:, 2].reshape(4283, 1)], 1)

        X = X.long()
        # 判断的两个蛋白质
        drugA = X[:, 0]
        drugB = X[:, 1]
        finalEmbedding = torch.cat([Embedding[drugA], Embedding[drugB]], 1).float()
        return self.fullConnectionLayer(finalEmbedding)
