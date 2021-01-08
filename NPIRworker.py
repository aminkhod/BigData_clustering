
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


from NPIR import NPIR

from sklearn import metrics

# import matplotlib.pyplot as plt

# ### multiprocessing
# from multiprocessing.pool import Pool
# import multiprocessing


import datetime
# import warnings
from collections import Counter as Cs
from timeit import default_timer as timer
from datetime import timedelta

import findspark
findspark.init()

from pyspark import SparkFiles


# In[1]:


def NPIRPreProcess():
    self.points = points
    #IR: The indexing ratio to be used for generating the maximum index
    self.IR = 0.2
    #The number of iteration i
    self.i = 10
    self.k = 3 #k: Number of clusters
    # count = Cs()
    self.chunk = 200
    leaderheadr = ['chunkLabel', 'old label']
    # leaderheadr = []
    leaderheadr.extend([str(x) for x in range(1, len(data_spark.columns))])
    leaderheadr = tuple(leaderheadr)

    start = timer()
    # labels = sqlContext.createDataFrame([np.full(len(labelsheader), np.nan).tolist()],labelsheader)
    # labels = labels.na.drop()

    leaders = sqlContext.createDataFrame([np.full(len(leaderheadr), np.nan).tolist()],leaderheadr)
    leaders = leaders.na.drop()

    ii = 0
    for z in range(0, points, chunk):
        j = z + chunk
        data = data_spark.where(col("index_column_name").between(z, j-1)).toPandas()
        data.drop("index_column_name",axis=1,inplace=True)
        data = data.astype(float)
        from NPIR import NPIR
        label = NPIR(data.values,k,IR,i)

        del NPIR
        data['labels'] = label

    #     # Adding to pyspard label
    #     chunklabel = np.full(len(label), ii).tolist()
    #     labelDF = [(x, y) for x, y in zip(chunklabel, label)]
    #     labelsDF = sqlContext.createDataFrame(labelDF,labelsheader)
    #     labels = unionAll(labels, labelsDF)

        leader = []
        f = list(Cs(label))
        f.sort()
        for i in f:
            leader.append([round(np.mean(z), 4) for z in data[data['labels']==i].values[:, :-1].T])
        del data

        # Adding to pyspark leaders
        for x in range(len(leader)):
            x1 = [ii, x]
            x1.extend(leader[x])
            leader[x] = x1
        leaderDF = sqlContext.createDataFrame(leader,leaderheadr)
        leaders = unionAll(leaders, leaderDF)
        ii += 1
    del data_spark
    end = timer()
    print ("Execution time HH:MM:SS:", timedelta(seconds= end - start))

