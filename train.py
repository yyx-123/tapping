import torch
import torch.nn as nn
import torch.nn.functional as F
from models import GCN, GraphSAGE, ChebNet, GAT, ARMA, SuperGAT
import utils
from torch_geometric.data import DataLoader
import numpy as np
import time

def loadData(freqSel=254, onlyOxy=True):
    #数据集加载、划分
    if onlyOxy:
        dataset = torch.load(".\dataset\\3_1dataLists_Oxy\dataList_" + str(freqSel) + ".pt")
    else:
        dataset = torch.load(".\dataset\\3_2dataLists_Oxy_DeOxy\dataList_" + str(freqSel) + ".pt")

    TRAIN_PERCENT = 0.8
    TEST_PERCENT = 0.1
    VALIDATION_PERCENT = 0.1
    trainSize = int(len(dataset) * TRAIN_PERCENT)
    testSize = int(len(dataset) * TEST_PERCENT)
    validationSize = len(dataset) - trainSize - testSize
    train_dataset, test_dataset, validation_dataset = torch.utils.data.random_split(dataset, [trainSize, testSize, validationSize])

    BATCH_SIZE = 128
    train_dataLoader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataLoader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validation_dataLoader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True)

    return train_dataLoader, test_dataLoader, validation_dataLoader, freqSel

def train(modelName, train_dataLoader, test_dataLoader, validation_dataLoader, freqSel, usePooling=True):
    # 模型参数定义
    DROPOUT = 0.3
    N_HIDDEN = 128
    N_OUT = 32
    LEARNING_RATE = 0.002
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 网络加载
    if modelName == "GCN":
        model = GCN(
            nfeat=train_dataLoader.dataset[0].num_node_features,
            nhid=N_HIDDEN,
            nout=N_OUT,
            dropout=DROPOUT,
            usePooling=usePooling
        ).to(device)
    elif modelName == "GraphSAGE":
        model = GraphSAGE(
            nfeat=train_dataLoader.dataset[0].num_node_features,
            nhid=N_HIDDEN,
            nout=N_OUT,
            dropout=DROPOUT,
            usePooling=usePooling
        ).to(device)
    elif modelName == "ChebNet":
        model = ChebNet(
            nfeat=train_dataLoader.dataset[0].num_node_features,
            nhid=N_HIDDEN,
            nout=N_OUT,
            dropout=DROPOUT,
            usePooling=usePooling
        ).to(device)
    elif modelName == "GAT":
        model = GAT(
            nfeat=train_dataLoader.dataset[0].num_node_features,
            nhid=N_HIDDEN,
            nout=N_OUT,
            dropout=DROPOUT,
            usePooling=usePooling
        ).to(device)
    elif modelName == "ARMA":
        model = ARMA(
            nfeat=train_dataLoader.dataset[0].num_node_features,
            nhid=N_HIDDEN,
            nout=N_OUT,
            dropout=DROPOUT,
            usePooling=usePooling
        ).to(device)
    elif modelName == "SuperGAT":
        model = SuperGAT(
            nfeat=train_dataLoader.dataset[0].num_node_features,
            nhid=N_HIDDEN,
            nout=N_OUT,
            dropout=DROPOUT,
            usePooling=usePooling
        ).to(device)
    else:
        print("No such model!")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    #网络训练
    EPOCH_NUM = 50
    accs = []
    for epoch in range(EPOCH_NUM):
        model.train()
        for train_data in train_dataLoader:
            train_data = train_data.to(device)
            out = model(train_data)
            loss_train = F.nll_loss(out, train_data['y'])
            # Backpropagation
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

        model.eval()
        correct_cnt = 0
        confusion = np.zeros([3, 3])
        for test_data in test_dataLoader:
            test_data = test_data.to(device)
            out = model(test_data)
            pred = torch.max(out, dim=1).indices
            for idx in range(len(test_data['y'])):
                # 使用混淆矩阵计算
                confusion[test_data['y'][idx]][pred[idx]] += 1
                # 三分类
                # if pred[idx] == test_data['y'][idx]:
                #     correct_cnt += 1

        # 计算各项分类指标
        # 准确率：混淆矩阵对角线元素和（分类正确的样本数） / 总样本数
        acc = (confusion[0][0] + confusion[1][1] + confusion[2][2]) / len(test_dataLoader.dataset)
        accs.append(acc)
        # 查准率：对角线元素（正确分类的样本数） / 该列的元素和（所有被分为该类的样本数）
        prec0, prec1, prec2 = utils.calPrecision(confusion)
        # 查全率
        recall0, recall1, recall2 = utils.calRecall(confusion)

        # 输出结果
        print('Epoch:{} | loss:{:.2f} | acc:{:.4f} | prec0:{:.4f} | prec1:{:.4f} | prec2:{:.4f} | recall0:{:.4f}  | recall1:{:.4f} | recall2:{:.4f}'\
            .format(epoch, loss_train, acc, prec0, prec1, prec2, recall0, recall1,recall2))
        # 保存特别好的模型
        # if acc > 0.6:
        #     localTime = time.localtime(time.time())
        #     fileName = "acc_" + str(int(acc * 1000)) + "_model254_" + modelName + "_time_" + str(localTime.tm_mon) + "_" + str(localTime.tm_mday) + "_" + str(localTime.tm_hour) + "_" + str(localTime.tm_min)
        #     torch.save(model.state_dict(), "./savedModels/Oxy Dxy models/" + fileName)



    print(np.mean(accs[-10 : -1]))


if __name__ == "__main__":
    train_dataLoader, test_dataLoader, validation_dataLoader, freqSel = loadData(onlyOxy=True, freqSel=253)
    for i in range(5):
        train(modelName="ChebNet",train_dataLoader=train_dataLoader, test_dataLoader=test_dataLoader, validation_dataLoader=validation_dataLoader, freqSel=freqSel)
