import torch
from torch_geometric.data import Data
import os
from scipy.stats import pearsonr
import numpy as np
import pywt
import matplotlib.pyplot as plt
from tqdm import tqdm

def getData(freqSel):
    srcDir = "dataset\\2_3splited_Dxy\\"
    fileList = os.listdir(srcDir)
    dataList = []
    for file in tqdm(fileList):
        task = file.split("_")[-1]
        nirs = np.loadtxt(srcDir + file)

        # 计算通道之间的相干性
        cor = np.zeros([20, 20])
        for i in range(19):
            for j in range(i + 1, 20):
                cor[i][j] = abs(pearsonr(nirs[i], nirs[j])[0])

        # 计算邻接矩阵和边特征矩阵（边权重）
        THRESHOLD = 0.2
        fromNode = []
        toNode = []
        edge_attr = []
        for i in range(19):
            for j in range(i + 1, 20):
                if cor[i][j] >= THRESHOLD:
                    fromNode.append(i)
                    toNode.append(j)
                    fromNode.append(j)
                    toNode.append(i)
                    edge_attr.append(cor[i][j])
                    edge_attr.append(cor[i][j])

        edge_index = torch.tensor([fromNode, toNode], dtype=torch.int64)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32)

        # 计算节点特征矩阵X（用小波变换）
        X = []
        for i in range(20):
            # 读取通道数据并截取START_SECOND秒后的部分
            START_SECOND = 2  # 由于HRF响应存在一定的延迟，这里设定延迟时间
            chDxyData = nirs[i][int(START_SECOND / 0.075) : -1]

            # 计算该通道的特征
            feature = []
            feature.append(0 if i <= 9 else 1)      # 第1维特征：左右脑区。左脑区为0右脑区为1
            feature.append(np.mean(chDxyData))    # 第2维特征：信号均值
            feature.append(np.var(chDxyData))     # 第3维特征：信号方差
            feature.append(max(chDxyData) - min(chDxyData)) # 第4维特征：信号幅值
            # 第5维特征：Dxy的小波系数。长度为信号的长度，以下的cwtmatr[253]的含义是研究频率约为0.07Hz左右的小波系数，可以根据需要修改
            [cwtmatr, frequencies] = cwt(chDxyData)
            feature = feature + list(cwtmatr[freqSel])

            # 特征标准化
            miu = np.mean(feature)
            std = np.std(feature)
            feature = (feature - miu) / std

            X.append(feature)

        X = np.array(X)
        X = torch.tensor(X, dtype=torch.float32)

        # target
        if task == "RHT":
            y = torch.tensor([0], dtype=torch.long)
        elif task == "LHT":
            y = torch.tensor([1], dtype=torch.long)
        else:
            y = torch.tensor([2], dtype=torch.long)

        # 封装该被试该次任务下的数据
        data = Data(x=X, edge_index=edge_index, edge_attr=edge_attr, y=y)
        dataList.append(data)

    # 保存所有数据到pt文件中
    torch.save(dataList, "dataset\\3_3dataLists_Dxy\dataList_" + str(freqSel) + ".pt")

def cwt(data, wavename="morl", fs = 13.333):
    t = np.arange(0, len(data)) / fs

    totalscale = 256
    fc = pywt.central_frequency(wavename)  # 中心频率
    cparam = 2 * fc * totalscale
    scales = cparam / np.arange(totalscale, 1, -1)
    [cwtmatr, frequencies] = pywt.cwt(data, scales, wavename, 1.0 / fs)  # 连续小波变换
    return [cwtmatr, frequencies]

def calRecall(confusion):
    recall0 = confusion[0][0] / confusion[0].sum()
    recall1 = confusion[1][1] / confusion[1].sum()
    recall2 = confusion[2][2] / confusion[2].sum()
    return recall0, recall1, recall2

def calPrecision(confusion):
    prec0 = confusion[0][0] / (confusion[0][0] + confusion[1][0] + confusion[2][0])
    prec1 = confusion[1][1] / (confusion[0][1] + confusion[1][1] + confusion[2][1])
    prec2 = confusion[2][2] / (confusion[0][2] + confusion[1][2] + confusion[2][2])
    return prec0, prec1, prec2


if __name__ == "__main__":
    getData(254)

