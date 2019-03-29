import numpy as np
import pandas as pd

def loss(w,b,X,y):
    return np.mean((np.dot(w,X.T)+b -y)**2)/2
def grad(w,b,X,y):
    grad_w = np.mean((np.dot(w,X.T)+b-y).T*X,axis=0)
    grad_b = np.mean(np.dot(w,X.T)+b-y)
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

# print(loss(w,b,train_x,train_y))
# grad_w,grad_b = grad(w,b,train_x,train_y)
# print(grad_w,grad_b)
# w,b = updat_grad(w,b,grad_w,grad_b,0.000001)
# print(w,b)
# print(loss(w,b,train_x,train_y))
for i in range(1000000):
    if i%50000==0: print(loss(w,b,train_x,train_y))
    grad_w,grad_b = grad(w,b,train_x,train_y)
    w,b = updat_grad(w,b,grad_w,grad_b,0.0001)

print("损失值：",loss(w,b,train_x,train_y))

np.save('w.npy',w)
np.save('b.npy',b)
