import numpy as np
from matplotlib import pyplot as plt
from random import shuffle
import csv
def data():
    '''
    加载数据，数据集为鸢尾花数据集120个训练集，30个测试集
    :return:
    '''
    with open('iris.data')as f:
        reader = csv.reader(f)
        data_list =list(reader)
    shuffle(data_list)
    X_orign = np.array([(list(map(float,x[:4]))) for x in data_list])
    y_orign = []
    for x in data_list:
        if x[4]=='Iris-setosa':
            y_orign.append(0)
        elif x[4]=='Iris-versicolor':
            y_orign.append(1)
        else:
            y_orign.append(2)
    y_orign = np.array(y_orign)
    X_train = X_orign[:120]
    y_train = y_orign[:120]
    X_test = X_orign[120:]
    y_test = y_orign[120:]
    return X_train,y_train,X_test,y_test
def plot_data(X,y):
    '''
    绘制数据集
    :param X:
    :param y:
    :return:
    '''
    plt.scatter(X[y==0,0],X[y==0,2],s=15,c='r',marker='o')
    plt.scatter(X[y==1,0],X[y==1,2],s=15,c='b',marker='+')
    plt.scatter(X[y==2,0],X[y==2,2],s=15,c='y',marker='*')
    plt.show()
def knn(x,K,X,y):
    '''
    采用线性扫描，求得与目标点x在X中前K个最近的距离
    :param x:
    :param K:
    :param X:
    :param y:
    :return:
    '''
    y =  y[np.argpartition(np.sum((X-x)**2,axis=1),K)[:K]]#欧式距离，获得前K小个元素的索引位置
    return np.argmax([sum(y==0),sum(y==1),sum(y==2)])#返回所属类别，少数服从多数

K = 5
X_train,y_train,X_test,y_test = data()
predict = np.array([knn(i,K,X_train,y_train) for i in X_test])#预测测试集每个元素所属的类别
print('正确率为：{}%'.format(sum(predict==y_test)/len(y_test)*100))#计算正确率
