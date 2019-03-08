import numpy as np
from matplotlib import pyplot as plt
def data(num):
    '''
    生成线性可分数据集，分界线为x0-x1+100=0
    :param num:
    :return:
    '''
    X = np.random.randint(0,101,(2,num))
    y = np.zeros(num)
    y[X[0,:]>=(100-X[1,:])] = 1
    y[X[0,:]<(100-X[1,:])] = -1
    return X,y
def plot_data(X,y,w,b):
    '''
    绘制散点图
    :param X:
    :param y:
    :param w:
    :param b:
    :return:
    '''
    plt.scatter(X[0,y==1],X[1,y==1],s=15,c='b',marker='o')
    plt.scatter(X[0,y==-1],X[1,y==-1],s=15,c='g',marker='x')
    plt.plot([0,-b/w[0]],[-b/w[1],0],c='r')
    plt.show()
def mistake(X,y,w,b):
    '''
    找到错误分类
    :param X:
    :param y:
    :param w:
    :param b:
    :return:
    '''
    return y*(w.dot(X)+b)<0
def compute_cost(X,y,w,b,c):
    '''
    计算损失值
    :param X:
    :param y:
    :param w:
    :param b:
    :param c:
    :return:
    '''
    return -np.sum(y[c]*(w.dot(X[:,c])+b))
def grad(X,y,c):
    '''
    计算梯度
    :param X:
    :param y:
    :param c:
    :return:
    '''
    dw = -np.sum(y[c]*X[:,c],axis=1)
    db = -np.sum(y[c])
    return dw,db
def updata_parameters(w,b,dw,db,learning_rate):
    '''
    梯度下降，更新参数
    :param w:
    :param b:
    :param dw:
    :param db:
    :param learning_rate:
    :return:
    '''
    w = w-learning_rate*dw
    b = b-learning_rate*db*100
    return w,b
#生成线性可分数据集，初始化参数
X,y = data(50)
w = np.random.rand(2)
b = 0
#计算错误分类点，求损失值
c = mistake(X,y,w,b)
cost = compute_cost(X,y,w,b,c)
i = 0
#当有错位分类点不停迭代
dw,db = grad(X,y,c)#计算梯度
print(dw)
w,b = updata_parameters(w,b,dw,db,0.001)#更新参数
while cost>0:
    if i%50==0:
        print('迭代{0}次，损失值为{1}:'.format(i,cost))
    dw,db = grad(X,y,c)#计算梯度
    w,b = updata_parameters(w,b,dw,db,0.001)#更新参数
    c = mistake(X,y,w,b)#求错误分类点，bool数组形式
    cost = compute_cost(X,y,w,b,c)#计算损失值
    i = i+1
#绘制图像，打印参数
plot_data(X,y,w,b)
print(w,b)

