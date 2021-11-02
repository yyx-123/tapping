from libsvm.svmutil import *
from libsvm.commonutil import *
from sklearn.model_selection import train_test_split

# 该脚本用以分析得到的实验数据

# 该函数通过SVM分析节点的特征矩阵，计算SVM的准确率，用以与GNN模型相对比
def SVM_checkGNN_nodeAttr():
    # 加载数据
    y, x = svm_read_problem('.\dataset\datasetForSVM\datasetForSVM_NodeAttr.txt')

    trainsetPercent = [0.8]

    for p in trainsetPercent:

        for i in range(5):
            print("---------------------------------------------------------------------------")
            print("p = {} , i = {} ".format(p, i))

            # 将数据集分割成训练集和测试集
            x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=p, shuffle=True)

            # 训练模型
            print("Start training....")
            model = svm_train(y_train, x_train, '-t 0 -c 4 -q')

            # 测试模型精度
            print("Start predicting....")
            p_label, p_acc, p_val = svm_predict(y_test, x_test, model)



if __name__ == "__main__":
    SVM_checkGNN_nodeAttr()