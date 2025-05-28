import torch
import torch.nn as nn
from torch import utils
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold  #交叉验证方法
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import logging
import math
import Encoder
import Fusion
import GAT
import Tool



# train(net, train_iter, num_epochs, lr, wd, X_test, y_test, i)
# 在定义一个名为 train 的函数时，您提供了一个用于训练神经网络模型的框架。
# 这个函数接受多个参数，包括网络模型 net、训练数据迭代器 train_iter、训练轮次 num_epochs、学习率 lr、权重衰减（也称为正则化项）wd、
# 测试数据集 X_test 和 y_test、折叠索引 fold（可能用于交叉验证中），以及设备 device 用于指定模型和数据应在哪种计算设备上运行（如CPU或GPU）。
def train(net, train_iter, num_epochs, lr, wd, X_test, y_test, fold=0, device=torch.device(f'cuda')):
    net.to(device)
    # 在PyTorch中，torch.optim.Adam 是一个优化器，它实现了Adam算法，这是一种基于梯度下降的优化算法，用于更新神经网络中的权重。
    # net.parameters()：这是一个生成器，它返回模型net中所有可训练参数的迭代器。这些参数是优化器需要更新的对象。
    # lr=lr：这里lr是学习率（Learning Rate）的缩写，它控制了权重更新的步长大小。
    # weight_decay=wd：weight_decay是正则化项的一个参数，用于防止模型过拟合。
    # 在Adam优化器中，它通常通过向梯度中添加一个与权重成正比的项来实现，这个项会倾向于减小权重的绝对值。wd是权重衰减的系数，其值也是在调用train函数时传入。
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    # nn.BCELoss 是二元交叉熵损失（Binary Cross Entropy Loss）的一个实现，它通常用于二分类问题中
    # 如果你有一个二分类问题，并且你的网络最后一层是线性的（即输出未经sigmoid处理的logits），
    # 你应该在将输出传递给 nn.BCELoss 之前，使用sigmoid函数（可以通过 torch.sigmoid 函数实现）来处理这些输出
    loss = nn.BCELoss()
    train_acc_list = []
    test_acc_list = []
    max_test_acc = 0
    max_test_sen = 0
    max_test_pre = 0
    max_test_auc = 0
    max_test_aupr = 0
    max_test_mcc = 0
    for epoch in range(num_epochs):
        train_result = []
        net.train()
        for i, (X, y) in enumerate(train_iter):
            optimizer.zero_grad()  # 清除之前累积的梯度
            X, y = X.to(device), y.to(device)
            y_hat = net(X)  # 预测值
            l = loss(y_hat, y.float())  # 计算损失值
            l.backward()  # 反向传播
            optimizer.step()  # 根据计算出的梯度更新模型的参数
            # 累加模型预测正确的数量
            train_result += torch.flatten(torch.round(y_hat).int() == y.int()).tolist()

        # 将网络（模型）设置为评估模式（evaluation mode）
        net.eval()

        # with torch.no_grad(): 是一个上下文管理器（context manager），它用于暂时禁用梯度计算
        with torch.no_grad():
            y_hat = net(X_test)

            tn, fp, fn, tp = confusion_matrix(y_test, torch.round(y_hat).int().cpu()).ravel()  #tn, fp, fn, tp
            # 计算接收者操作特征曲线（ROC curve）的假正率（FPR）、真正率（TPR，也称为召回率或灵敏度）以及对应的阈值（thresholds）
            fpr, tpr, thresholds = metrics.roc_curve(y_test, y_hat.cpu(), pos_label=1)
            # 计算二分类问题的精确率（Precision）、召回率（Recall，也称为真正率TPR）以及对应的阈值（Thresholds）
            precision, recall, thresholds = metrics.precision_recall_curve(y_test, y_hat.cpu())

            train_acc = sum(train_result) / len(train_result)
            test_acc = (tn + tp) / (fp + tp + fn + tn)
            test_sen = tp / (tp + fn)  # sensitivity = TP / (TP + FN)
            test_pre = tp / (tp + fp)
            test_auc = metrics.auc(fpr, tpr)
            test_aupr = metrics.auc(recall, precision)
            test_mcc = (tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            if test_acc == max(test_acc_list):
                max_test_acc = test_acc
                max_test_sen = test_sen
                max_test_pre = test_pre
                max_test_auc = test_auc
                max_test_aupr = test_aupr
                max_test_mcc = test_mcc
            print(f'tn:{tn}, fp:{fp}, fn:{fn}, tp:{tp}')
            print(f'train acc {train_acc:.4f}')
            print(f'test acc {test_acc:.4f}')
            print(f'test sen {test_sen:.4f}')
            print(f'test pre {test_pre:.4f}')
            print(f'test auc {test_auc:.4f}')
            print(f'test aupr {test_aupr:.4f}')
            print(f'test mcc {test_mcc:.4f}')
            print(f'max test acc {max_test_acc:.4f}')
            print(f'max test sen {max_test_sen:.4f}')
            print(f'max test pre {max_test_pre:.4f}')
            print(f'max test auc {max_test_auc:.4f}')
            print(f'max test aupr {max_test_aupr:.4f}')
            print(f'max test mcc {max_test_mcc:.4f}')
            print("epoch:{}...".format(epoch))
            print("fold:{}...".format(fold))
            print("----------------------------------------------")
            logging.info(f'train acc {train_acc:.4f}')
            logging.info(f'test acc {test_acc:.4f}')
            logging.info(f'test sen {test_sen:.4f}')
            logging.info(f'test pre {test_pre:.4f}')
            logging.info(f'test auc {test_auc:.4f}')
            logging.info(f'test aupr {test_aupr:.4f}')
            logging.info(f'test mcc {test_mcc:.4f}')
            logging.info(f'max test acc {max_test_acc:.4f}')
            logging.info(f'max test sen {max_test_sen:.4f}')
            logging.info(f'max test pre {max_test_pre:.4f}')
            logging.info(f'max test auc {max_test_auc:.4f}')
            logging.info(f'max test aupr {max_test_aupr:.4f}')
            logging.info(f'max test mcc {max_test_mcc:.4f}')
            logging.info("epoch:{}...".format(epoch))
            logging.info("fold:{}...".format(fold))
            logging.info("----------------------------------------------")

    # x = range(len(train_acc_list))
    # y1 = train_acc_list
    # y2 = test_acc_list
    # plt.plot(x, y1, color='r', label="train_acc")  # s-:方形
    # plt.plot(x, y2, color='g', label="test_acc")  # o-:圆形
    # plt.xlabel("epoch")  # 横坐标名字
    # plt.ylabel("accuracy")  # 纵坐标名字
    # plt.legend(loc="best")  # 图例
    # plt.savefig('../data/' + logName + str(epoch) + '.png')

    return max_test_acc, max_test_sen, max_test_pre, max_test_auc, max_test_aupr, max_test_mcc


# 学习率 权重衰减(l2正则化) PAN PPI 数据集被分割成多少个子集 被用于更新模型权重的次数（训练轮次） 一次梯度更新中所处理的样本数量
def train_KFold(lr, wd, KG1, data, n_splits, num_epochs, batch_size):
    test_acc_list = []
    test_sen_list = []
    test_pre_list = []
    test_auc_list = []
    test_aupr_list = []
    test_mcc_list = []

    # 创建StratifiedKFold对象
    # 在PPI网络中，80%的三元组作为训练集，剩下的20%作为测试集。
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    features = data.iloc[:, 0:2]  # (蛋白质、蛋白质) 前两列作为特征
    labels = data.iloc[:, 2:]  # 关系 第三列

    # 进行交叉验证
    # kf.split(features, labels)：这个方法根据features（特征）和labels（标签）来生成训练集和测试集的索引。
    # 它会返回一系列的元组，每个元组包含两个数组：一个用于训练集的索引，另一个用于测试集的索引。
    # enumerate(...)：这个函数用于将kf.split(...)返回的元组序列转换成一个包含索引和元组值的迭代器。
    # 这样，在循环中，i将是迭代次数（从0开始），而(train_index, test_index)将是当前迭代的训练集和测试集索引。
    for i, (train_index, test_index) in enumerate(kf.split(features, labels)):
        X_train, X_test = features.iloc[train_index], features.iloc[test_index]
        y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]

        # 将 X_train 的第二列（索引为 1）转换成一个列表，并将其作为字典 X_train2 中键 '0' 的值。
        X_train2 = {'0': list(X_train.iloc[:, 1]), '1': list(X_train.iloc[:, 0])}
        X_train2 = pd.DataFrame(data=X_train2)

        # 将 X_train2 添加到 X_train 的行下面，形成一个新的 DataFrame，并通过 reset_index(drop=True) 重置索引，删除旧的索引并创建一个新的从 0 开始的整数索引。
        # 将 X_train的两列蛋白质关系交换顺序，添加到后面行，关系是一样的直接复制添加到下面
        X_train = pd.concat([X_train, X_train2], axis=0).reset_index(drop=True)
        y_train = pd.concat([y_train, y_train], axis=0).reset_index(drop=True)

        # X_train, y_train 列合并
        KG2 = torch.tensor(pd.concat([X_train, y_train], axis=1).to_numpy()).to('cuda')

        X_train = torch.tensor(X_train.to_numpy())
        y_train = torch.tensor(y_train.to_numpy())
        X_test = torch.tensor(X_test.to_numpy())
        y_test = torch.tensor(y_test.to_numpy())

        # 创建一个TensorDataset对象，将训练数据的特征和标签封装在一起
        dataset = utils.data.TensorDataset(X_train, y_train)
        # train_iter是一个DataLoader对象，它接受dataset作为输入，并设置批处理大小batch_size以及是否打乱数据shuffle=True。
        train_iter = utils.data.DataLoader(dataset, batch_size, shuffle=True)

        # 使用PyTorch框架中的nn.Sequential容器来构建一个神经网络模型。
        # nn.Sequential是一个顺序容器，用于按顺序包装一系列层。在这个例子中，模型由三个主要层组成：两个GNNLayer层和一个FusionLayer层
        net = nn.Sequential(
            # 第一个GNNLayer层：PAN
            # 输入维度：net_args.kgDimension，PAN的特征维度。
            # 采样数量：net_args.sample_number，表示在每个节点上采样的邻居数量。
            # 知识图谱：KG1，这是第一个知识图谱的实例或表示。PAN
            # 传播层数：net_args.pLayers，表示图神经网络中的传播层数。
            # 索引：0，可能用于标识或区分不同的GNNLayer。
            GAT.GATLayer(Tool.net_args.kgDimension, Tool.net_args.sample_number, KG1, 0),
            # 第二个GNNLayer层：PPI
            # 输入维度：net_args.ddiDimension，PPI的特征维度。
            # 采样数量、传播层数与第一个GNNLayer相同。
            # 知识图谱：KG2，这是第二个知识图谱的实例或表示。PPI
            # 索引：1，用于标识或区分。
            # True：
            GAT.GATLayer(Tool.net_args.ddiDimension, Tool.net_args.sample_number, KG2, 1),
            # KG1蛋白质-实体全部，KG2蛋白质-蛋白质训练集   AUTO_REC层
            Encoder.EncoderLayer(Tool.net_args.kgDimension_enCode, KG1, 0),
            Encoder.EncoderLayer(Tool.net_args.ddiDimension_enCode, KG2, 1),
            # FusionLayer层：
            # 输入维度：net_args.kgDimension + net_args.ddiDimension，这是前两个GNNLayer层输出维度的和，表示融合层的输入特征维度。
            # net_args.dropout：表示在融合层中应用的dropout比率，用于减少过拟合。
            # 索引：2，用于标识或区分。
            Fusion.FusionLayer(
                Tool.net_args.kgDimension + Tool.net_args.ddiDimension + Tool.net_args.kgDimension_enCode + Tool.net_args.ddiDimension_enCode,
                Tool.net_args.dropout, 4))
        test_acc, test_sen, test_pre, test_auc, test_aupr, test_mcc = \
            train(net, train_iter, num_epochs, lr, wd, X_test, y_test, i)
        # train(net, train_iter, num_epochs, lr, wd, X_test, y_test, fold=0, device=torch.device(f'cuda'))

        test_acc_list.append(test_acc)
        test_sen_list.append(test_sen)
        test_pre_list.append(test_pre)
        test_auc_list.append(test_auc)
        test_aupr_list.append(test_aupr)
        test_mcc_list.append(test_mcc)
        print(f'fold {i},  max_test_acc:{test_acc:.4f}')
        print(f'fold {i},  max_test_sen:{test_sen:.4f}')
        print(f'fold {i},  max_test_pre:{test_pre:.4f}')
        print(f'fold {i},  max_test_auc:{test_auc:.4f}')
        print(f'fold {i},  max_test_aupr:{test_aupr:.4f}')
        print(f'fold {i},  max_test_mcc:{test_mcc:.4f}')
        print("---------------------------------------")
        logging.info(f'fold {i},  max_test_acc:{test_acc:.4f}')
        logging.info(f'fold {i},  max_test_sen:{test_sen:.4f}')
        logging.info(f'fold {i},  max_test_pre:{test_pre:.4f}')
        logging.info(f'fold {i},  max_test_auc:{test_auc:.4f}')
        logging.info(f'fold {i},  max_test_aupr:{test_aupr:.4f}')
        logging.info(f'fold {i},  max_test_mcc:{test_mcc:.4f}')
        logging.info("---------------------------------------")
    print(f'avg test acc {sum(test_acc_list) / n_splits:.4f}')
    print(f'avg test sen {sum(test_sen_list) / n_splits:.4f}')
    print(f'avg test pre {sum(test_pre_list) / n_splits:.4f}')
    print(f'avg test auc {sum(test_auc_list) / n_splits:.4f}')
    print(f'avg test aupr {sum(test_aupr_list) / n_splits:.4f}')
    print(f'avg test mcc {sum(test_mcc_list) / n_splits:.4f}')
    logging.info(f'avg test acc {sum(test_acc_list) / n_splits:.4f}')
    logging.info(f'avg test sen {sum(test_sen_list) / n_splits:.4f}')
    logging.info(f'avg test pre {sum(test_pre_list) / n_splits:.4f}')
    logging.info(f'avg test auc {sum(test_auc_list) / n_splits:.4f}')
    logging.info(f'avg test aupr {sum(test_aupr_list) / n_splits:.4f}')
    logging.info(f'avg test mcc {sum(test_mcc_list) / n_splits:.4f}')
