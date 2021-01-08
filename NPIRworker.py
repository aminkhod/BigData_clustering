
# coding: utf-8

# In[4]:


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

from operator import *


from pyspark import SparkContext
# from pyspark import StorageLevel
from pyspark.sql.types import *
from pyspark.sql.functions import udf, log, rand, col, broadcast, row_number, avg, mean, least, struct,                lit, sequence, sum, monotonically_increasing_id, pandas_udf, PandasUDFType
import pyspark.sql.functions as F

from functools import reduce
from pyspark.sql import SparkSession, SQLContext, Window, Row, DataFrame
from pyspark import SparkConf
from scipy.spatial import distance
from pyspark.sql.window import Window

spark = SparkSession.builder.master("local[*]").config("spark.storage.blockManagerSlaveTimeoutMs","12000001ms").config("spark.driver.maxResultSize","24g").config("spark.default.parallelism", "200").config("spark.memory.offHeap.size", "24g").appName("NPIR_Parallel").config("spark.executor.memory", "24g").config("spark.driver.memory", "24g").getOrCreate()
# spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "False")
sc = spark.sparkContext
sqlContext = SQLContext(sc)


# In[6]:


def NPIRPreProcess(points, chunk):
    
    #IR: The indexing ratio to be used for generating the maximum index
    IR = 0.2
    #The number of iteration i
    i = 10
    k = 3 #k: Number of clusters
    # count = Cs()
#     leaderheadr = ['chunkLabel', 'old label']
#     # leaderheadr = []
#     leaderheadr.extend([str(x) for x in range(1, len(data_spark.columns))])
#     leaderheadr = tuple(leaderheadr)

    start = timer()
    # labels = sqlContext.createDataFrame([np.full(len(labelsheader), np.nan).tolist()],labelsheader)
    # labels = labels.na.drop()

#     leaders = sqlContext.createDataFrame([np.full(len(leaderheadr), np.nan).tolist()],leaderheadr)
#     leaders = leaders.na.drop()

#     ii = 0
    print(SparkFiles.get('blobs1.csv'))
    data_spark_df = spark.read.format('csv').option('header','True').option('index','False').    load(SparkFiles.get(some_path))
    for z in range(0, points, chunk):
        j = z + chunk

        data = data_spark_df.where(col("index_column_name").between(z, min(points, j-1))).toPandas()
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

#         # Adding to pyspark leaders
#         for x in range(len(leader)):
#             x1 = [ii, x]
#             x1.extend(leader[x])
#             leader[x] = x1
#         leaderDF = sqlContext.createDataFrame(leader,leaderheadr)
#         leaders = unionAll(leaders, leaderDF)
#         ii += 1
#     del data_spark
    return leader
    end = timer()
    print ("Execution time HH:MM:SS:", timedelta(seconds= end - start))

