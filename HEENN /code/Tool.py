import pandas as pd
import numpy as np
import random
import argparse
import warnings

warnings.simplefilter(action="ignore", category=pd.errors.SettingWithCopyWarning)
# 创建一个ArgumentParser对象(解释器)
parser = argparse.ArgumentParser(description='模型参数设置')

# 添加参数
parser.add_argument('--weight_decay', type=float, default=1e-4, choices=[3e-4], help="weight_decay")  # 权重衰减
parser.add_argument('--lr', type=float, default=5 * 1e-4, choices=[5e-4], help="learning rate")  # 学习率
parser.add_argument('--epoch', type=int, default=40, choices=[40], help="epochs_number")  # 整个训练数据集被用于更新模型权重的次数
parser.add_argument('--batch_size', type=int, default=8192, choices=[8192], help="batch_size")  # 模型在一次梯度更新中所处理的样本数量
parser.add_argument('--kgDimension', type=int, default=256, choices=[64], help="7kDimension")  # PAN特征维度
parser.add_argument('--ddiDimension', type=int, default=512, choices=[512], help="ddiDimension")  # PPI特征维度
parser.add_argument('--sample_number', type=int, default=60, choices=[7], help="sample_number")  # 每个节点上采样的邻居数量
parser.add_argument('--dropout', type=float, default=0.3, choices=[0.3], help="dropout")  # Dropout率（即神经元被丢弃的概率）
parser.add_argument('--pLayers', type=int, default=1, choices=[1], help="pLayers")  # 图神经网络中的传播层数
parser.add_argument('--n_splits', type=int, default=5, choices=[5], help="n_splits")  # 指定了数据集被分割成多少个子集
parser.add_argument('--kgDimension_enCode', type=int, default=64, choices=[64], help="7kDimension")  # PAN特征维度
parser.add_argument('--ddiDimension_enCode', type=int, default=64, choices=[512], help="ddiDimension")  # PPI特征维度

# 解析命令行参数
net_args = parser.parse_args()


# 转换PAN的格式
def data_preprocessing(data):
    protein_number = pd.read_csv("/home/jy/桌面/HEENN/KGF-GNN/KGF-GNN/data/ProteinNumber.csv")
    key_list = list(protein_number.iloc[:, 0])  # 取第0列所有行，没有列索引的标号
    value_list = list(protein_number.iloc[:, 1])
    # zip函数将这两个列表中的元素一一对应起来，形成一个元组的列表，每个元组包含一个键和一个值。然后，dict函数将这个元组列表转换成字典。
    protein_map = dict(zip(key_list, value_list))

    # 从蛋白质与实体表中提取出实体字典
    entitymap = {}
    entitylist = np.array(data.iloc[:, 1]).tolist()  # 实体的名称列表
    for entity in entitylist:
        if entity not in entitymap:
            entitymap[entity] = len(entitymap)

    # 从蛋白质与实体表中提取出关系字典
    relationmap = {}
    relationlist = np.array(data.iloc[:, 2]).tolist()
    for relation in relationlist:
        if relation not in relationmap:
            relationmap[relation] = len(relationmap)

    # 将每一列中每个元素通过字典进行转换 转换乘数值
    data.iloc[:, 0] = data.iloc[:, 0].map(protein_map)
    data.iloc[:, 1] = data.iloc[:, 1].map(entitymap)
    data.iloc[:, 2] = data.iloc[:, 2].map(relationmap)

    return data


# PPI负样本的生成
def NegativeGenerate(KG, PPIlist):
    NegativeSamplelist = []
    NegativeSampleCounter = 0
    while NegativeSampleCounter < len(PPIlist):
        PPInumber1 = random.randint(0, KG['0'].nunique() - 1)
        PPInumber2 = random.randint(0, KG['0'].nunique() - 1)
        if PPInumber1 == PPInumber2:
            continue
        PPIpair = []
        PPIpair.append(PPInumber1)
        PPIpair.append(PPInumber2)
        flag = 0
        for pair in PPIlist:
            if PPIpair == pair:
                flag = 1
                break
        if flag == 1:
            continue
        for pair in NegativeSamplelist:
            if PPIpair == pair:
                flag = 1
                break
        if flag == 1:
            continue
        if flag == 0:
            NegativeSamplelist.append(PPIpair)
            NegativeSampleCounter = NegativeSampleCounter + 1
            print(f'NegativeGenerate:{len(PPIlist) - NegativeSampleCounter}')
    return pd.DataFrame(NegativeSamplelist)
