import torch
import pandas as pd
import logging
import time
import Tool
import HEENN_train

logName = 'HEENN'
if __name__ == '__main__':
    start_time = time.time()
    logging.basicConfig(level=logging.INFO, filename='../result/' + logName + '.log', filemode='w',
                        format="%(message)s")
    datadf = pd.read_csv("../data/KnowledgeGraph.csv")
    KG1 = Tool.data_preprocessing(datadf)

    # 转换为PyTorch张量，并将其移动到CUDA设备
    KG1 = torch.tensor(KG1.to_numpy().astype(int)).to('cuda')  # 蛋白质与实体之间的关系(蛋白质、实体、关系)

    PPI = pd.read_csv("../data/event_encode.csv")  # 蛋白质与蛋白质之间的关系(蛋白质、蛋白质、关系)

    # 学习率 权重衰减(l2正则化) PAN PPI 数据集被分割成多少个子集 被用于更新模型权重的次数 一次梯度更新中所处理的样本数量
    HEENN_train.train_KFold(Tool.net_args.lr, Tool.net_args.weight_decay, KG1, PPI, Tool.net_args.n_splits,
                            Tool.net_args.epoch,
                            Tool.net_args.batch_size)

    end_time = time.time()
    print("代码运行了 {:.2f} 秒".format(end_time - start_time))
    logging.info("代码运行了 {:.2f} 秒".format(end_time - start_time))
