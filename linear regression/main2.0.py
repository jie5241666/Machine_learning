import numpy as np
import pandas as pd

def loss(w,b,X,y):
    return np.mean((np.dot(w,X.T)+b -y)**2)/2
def sgd(w,b,x,yi):
    grad_w = (np.dot(w,x.T)+b-yi)*x
    grad_b = (np.dot(w,x.T)+b-yi)
    return grad_w,grad_b
def updat_grad(w,b,grad_w,grad_b,learning_rate):
    w = w-learning_rate*grad_w
    b = b-learning_rate*grad_b
    return w,b
# 处理数据导入数据
data = pd.read_csv('train.csv',encoding = 'unicode_escape')
# 读入PM2.5数据
data = data[data[data.columns[2]].str.contains('PM2.5')]
#删除了不用的前三列
data = data.drop([data.columns[0],data.columns[1],data.columns[2]],axis=1)
# 取每10个小时的数据作为训练集
data1 = np.array(data)
train_x = np.array([data1[:,i:i+9] for i in range(15)],dtype=float).reshape(-1,9)
train_y = np.array([data1[:,i+9] for i in range(15)],dtype=float).reshape(-1)

w = np.zeros((1,9))
b = 0

print(train_x.shape)
for i in range(30):
    for i in range(train_x.shape[0]):
        l = loss(w,b,train_x,train_y)
        if i%2000==0: print(l)
        if l<0.05:
            break
        grad_w,grad_b = sgd(w,b,train_x[i],train_y[i])
        w,b = updat_grad(w,b,grad_w,grad_b,0.00001)


print("损失值:",loss(w,b,train_x,train_y))


