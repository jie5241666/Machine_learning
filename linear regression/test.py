import numpy as np
import pandas as pd
data = pd.read_csv('test.csv')
data = data[data[data.columns[1]].str.contains('PM2.5')]
data = data.drop([data.columns[0],data.columns[1]],axis=1)
test_x = np.array(data,dtype=float)
w = np.load('w.npy')
b = np.load('b.npy')
y = np.dot(w,test_x.T)+b
y = y.reshape(-1)
y = pd.Series(y)
data1 = pd.read_csv('sampleSubmission.csv')

data1['value'] = y
print(data1.head())
data1.to_csv('sample.csv',index=0)
