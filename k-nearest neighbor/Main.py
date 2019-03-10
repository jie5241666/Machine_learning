import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from random import shuffle
import csv
with open('iris.data')as f:
    reader = csv.reader(f)
    data_list =list(reader)
shuffle(data_list)
X = np.array([(list(map(float,x[:4]))) for x in data_list])
y = [x[4] for x in data_list]


