
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
import pandas as pd
import csv
import numpy as np
k = 20

#generate data
data, y = make_blobs(n_samples=3000000, centers=k, random_state=30)
#print(data)
#print(type(data)) np.ndarray
#plt.scatter(data[:, 0], data[:, 1], s=5)
data_df = pd.DataFrame(data)
data_df.to_csv('test_data_12.csv', index = True, encoding = 'UTF-8')

'''
data = np.genfromtxt('test_data_10.csv', delimiter=',')
data = data[1:]
data = np.delete(data,[0] ,1)

print(type(data),'\n', data[0])

'''
#plt.scatter(data[:, 0], data[:, 1], s=50)

