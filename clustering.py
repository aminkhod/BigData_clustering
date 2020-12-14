
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


from NPIR import NPIR

from sklearn.utils import shuffle
from sklearn import datasets
from sklearn.datasets import make_blobs
from sklearn import metrics

import matplotlib.pyplot as plt

import datetime
import warnings
from collections import Counter as Cs
from timeit import default_timer as timer
from datetime import timedelta


import findspark
findspark.init()

########
from numpy import array
from math import sqrt
from pyspark.mllib.clustering import KMeans, KMeansModel
########

from pyspark import SparkContext
from operator import *
from pyspark import StorageLevel
from pyspark.sql.types import IntegerType, FloatType, BooleanType, StringType, StructType,StructField,ArrayType, DataType
from pyspark.sql.functions import udf, log, rand, monotonically_increasing_id, col, broadcast,greatest, desc, asc, row_number, avg, mean, least, struct, lit, sequence, sum
from functools import reduce
import pyspark.sql.functions as F
from pyspark.sql import SparkSession, SQLContext, Window, Row, DataFrame
from pyspark import SparkConf


# In[2]:


spark = SparkSession.builder.master("local[*]").config("spark.sql.broadcastTimeout", "30000s").config("spark.network.timeout","30000s").config("spark.executor.heartbeatInterval","12000000ms").config("spark.storage.blockManagerSlaveTimeoutMs","12000001ms").config("spark.driver.maxResultSize","12g").config("spark.default.parallelism", "100").config("spark.memory.offHeap.enabled","true").config("spark.memory.offHeap.size", "25g").appName("NPIR_Parallel").getOrCreate()


# In[3]:


sc = spark.sparkContext
sqlContext = SQLContext(sc)


# In[90]:


# x1,y1 = make_blobs(n_samples=100000, centers=3, n_features=2,
#                 random_state=1)
# data = pd.DataFrame(x1)
# data['label'] = y1
# data.head()


# In[91]:


# data = shuffle(data)
# data.head()


# In[92]:


# data.to_csv('blobs2.csv', index=False)


# In[4]:


#read csv
data_spark_df = spark.read.format('csv').option('header','True').option('index','False').load('blobs2.csv')
# data_spark_df.show()


# In[5]:


data_spark_df.describe()


# In[6]:


data_spark_df.printSchema()


# In[7]:


data_spark_df = data_spark_df.select('0', '1')


# In[8]:


new_name = ['first', 'second']
data_spark_rdd = data_spark_df.toDF(*new_name).rdd.filter(lambda x:x)
data_spark_df = data_spark_df.toDF(*new_name)


# In[9]:


spark.conf.set("spark.sql.debug.maxToStringFields", 1000)
spark.conf.set('spark.jars.packages','com.databricks:spark-cav_2.11')
spark.conf.set("spark.sql.parquet.compression.codec","gzip")
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "False")
sqlContext.setConf("spark.sql.shuffle.partitions", "100")


# In[10]:


# x1,y1 = make_blobs(n_samples=100, centers=3, n_features=2,
#                 random_state=1)


# In[11]:


points = data_spark_df.count()
points


# In[20]:


100000/250


# In[21]:


#IR: The indexing ratio to be used for generating the maximum index
IR = 0.2
#The number of iteration i
i = 10
k = 3 #k: Number of clusters
# count = Cs()
chunk = 250


# In[22]:


# start = timer()
# kmeans = KMeans(n_clusters=k, random_state=0).fit(x)
# end = timer()
# print ("Execution time HH:MM:SS:",timedelta(seconds=end-start))


# In[23]:


from pyspark.sql.functions import desc, row_number, monotonically_increasing_id
from pyspark.sql.window import Window

data_spark = data_spark_df.withColumn('index_column_name', row_number().                                          over(Window.orderBy(monotonically_increasing_id())) - 1)
# data_spark.show()


# In[24]:


def unionAll(*dfs):
    return reduce(DataFrame.unionAll, dfs)


# In[25]:


data_spark.columns


# In[26]:


leaderheadr = ['chunkLabel', 'old label', 'count']
leaderheadr.extend([str(x) for x in range(1, len(data_spark.columns))])
leaderheadr = tuple(leaderheadr)
leaderheadr


# In[27]:


labelsheader = ('chunkLabel', 'label')
labelsheader


# In[28]:


start = timer()


labels = sqlContext.createDataFrame([np.full(len(labelsheader), np.nan).tolist()],labelsheader)
labels = labels.na.drop()

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
    data['labels'] = label
    
    # Adding to pyspard label
    chunklabel = np.full(len(label), ii).tolist()
    labelDF = [(x, y) for x, y in zip(chunklabel, label)]
    labelsDF = sqlContext.createDataFrame(labelDF,labelsheader)
    labels = unionAll(labels, labelsDF)

    leader = []
    f = list(Cs(label))
    f.sort()
    for i in f:
        leader.append([round(np.mean(z), 4) for z in data[data['labels']==i].values[:,:-1].T])
    del data
    del NPIR
    
    # Adding to pyspark leaders
    for x in range(len(leader)):
        x1 = [ii, x, Cs(label)[x]]
        x1.extend(leader[x])
        leader[x] = x1
    leaderDF = sqlContext.createDataFrame(leader,leaderheadr)
    leaders = unionAll(leaders, leaderDF)
    ii += 1
end = timer()
print ("Execution time HH:MM:SS:",timedelta(seconds=end-start))


# In[29]:


# Cs(label)


# In[30]:


# leaders.show()


# In[31]:


# # Load and parse the data
# data = sc.textFile("data/mllib/kmeans_data.txt")
# parsedData = data.map(lambda line: array([float(x) for x in line.split(' ')]))
parsedData = leaders.select(['1', '2']).rdd.map(list)

start = timer()
# Build the model (cluster the data)
clusters = KMeans.train(parsedData, k, maxIterations=20, initializationMode="random")
end = timer()
print ("Execution time HH:MM:SS:",timedelta(seconds=end-start))


# In[ ]:


# Evaluate clustering by computing Within Set Sum of Squared Errors
# def error(point):
#     center = clusters.centers[clusters.predict(point)]
#     return sqrt(sum([x**2 for x in (point - center)]))

# WSSSE = parsedData.map(lambda point: error(point)).reduce(lambda x, y: x + y)
# print("Within Set Sum of Squared Error = " + str(WSSSE))


# In[ ]:


# clusters.centers


# In[ ]:


# # Save and load model
# clusters.save(sc, "KMeansModel")
# sameModel = KMeansModel.load(sc, "KMeansModel")


# In[ ]:


# from numpy import array
# from math import sqrt

# from pyspark.mllib.clustering import KMeans, KMeansModel

# # Load and parse the data
# data = sc.textFile("data/mllib/kmeans_data.txt")
# parsedData = data.map(lambda line: array([float(x) for x in line.split(' ')]))

# # Build the model (cluster the data)
# clusters = KMeans.train(parsedData, 2, maxIterations=10, initializationMode="random")

# # Evaluate clustering by computing Within Set Sum of Squared Errors
# def error(point):
#     center = clusters.centers[clusters.predict(point)]
#     return sqrt(sum([x**2 for x in (point - center)]))

# WSSSE = parsedData.map(lambda point: error(point)).reduce(lambda x, y: x + y)
# print("Within Set Sum of Squared Error = " + str(WSSSE))

# # Save and load model
# clusters.save(sc, "target/org/apache/spark/PythonKMeansExample/KMeansModel")
# sameModel = KMeansModel.load(sc, "target/org/apache/spark/PythonKMeansExample/KMeansModel")


# # c1x c2y c3z
# # /
# # (c1 + c2 + c3)
# 
# # sum = (c1 + c2 + c3)
# # leader[c1/sum*x +c2/sum*y+ c3/sum*z]
# 

# In[ ]:


# labels.show()


# In[ ]:


# leaders = leaders.select(['chunkLabel','old label','1','2'])


# In[ ]:


# data_spark = leaders.withColumn("first_numeric", leaders["1"].cast(FloatType()))
# data_spark = data_spark.withColumn("second_numeric", data_spark["2"].cast(FloatType()))
# data_spark = data_spark.drop('1').drop('2').drop('chunkLabel').drop('old label')
# # #### Compute weighted average
# # count = data_spark.select('count')
# # ####

# data_spark = data_spark.drop('1').drop('2').drop('chunkLabel').drop('old label').drop('count')
# data_spark


# In[ ]:


# count = count.withColumn('Cindex', row_number().\
#                                           over(Window.orderBy(monotonically_increasing_id())) - 1)
# count


# In[ ]:


# def MyCheckUpdate(a, b, c, d):
#     a = float(a)
#     b = float(b)
#     c = float(c)
#     d = float(d)
#     res = (a-c) + (b-d)
#     if res == 0:
#         return 1.0
#     return 0.0

# check_centroid = udf(lambda x,y,z,r: MyCheckUpdate(x,y,z,r), FloatType())

def squaree1(c,u,f,g):
    c = float(c)
    u = float(u)
    f = float(f)
    g = float(g)
    array1 = np.array([c,u])
    array2 = np.array([f,g])
    dist = np.linalg.norm(array1-array2)
    dist = dist.item()
    return dist
squaree_spark1 = udf(lambda x,y,z,r: squaree1(x,y,z,r), FloatType())
sqlContext.sql("SET spark.sql.autoBroadcastJoinThreshold = -1")


# In[ ]:



# c = -8
# u = -9
# f = -7
# g = -8
# array1 = np.array([c,u])
# array2 = np.array([f,g])
# dist = np.linalg.norm(array1-array2)
# dist 


# In[ ]:


# df_centroid = data_spark.sample(False, 0.4,seed = 0).limit(1).cache()

# new_name = ['x','y']
# df_centroid = df_centroid.toDF(*new_name)

# #just for first round
# #first time
# i = 0
# data_cent = data_spark.join(broadcast(df_centroid))
# data_cent1 = data_cent.withColumn(str(i),squaree_spark1(data_cent.columns[0],data_cent.columns[1],
#                                               data_cent.columns[2*i+2],data_cent.columns[2*i+3]))
# data_cent2 = data_cent1.drop(data_cent1.columns[i+2]).drop(data_cent1.columns[i+3])
# data_cent3 = data_cent2.withColumn('mindist',col(str(i)))
# data_cent4 = data_cent3.withColumn('mindist1',least(data_cent3.columns[i+2], col('mindist')))
# data_cent4 = data_cent4.drop('mindist')
# data_cent5 = data_cent4.withColumnRenamed('mindist1','mindist')
# next_selected = data_cent5.orderBy(desc('mindist')).limit(1).select(data_cent5.columns[0:2])#1:3

# df_centroid = df_centroid.union(next_selected)
# u = [str(i)+'x',str(i)+'y']
# next_selected = next_selected.toDF(*u)
# def initial_centroids(next_selected,data_cent_5_persist, i):
#     data_cent6 = data_cent_5_persist.join(broadcast(next_selected))
#     data_cent6 = data_cent6.withColumn(str(i),squaree_spark1(data_cent6.columns[0],data_cent6.columns[1],
#                                              data_cent6.columns[i+3],data_cent6.columns[i+4]))#+4 +5
#     data_cent6 = data_cent6.drop(data_cent6.columns[i+3]).drop(data_cent6.columns[i+4])#+4 +5
#     data_cent6 = data_cent6.withColumn('mindist1',least(data_cent6.columns[i+3], col('mindist')))#4
#     data_cent6 = data_cent6.drop('mindist')
#     data_cent6 = data_cent6.withColumnRenamed('mindist1','mindist')
#     next_cent = data_cent6.orderBy(desc('mindist')).limit(1).select(data_cent6.columns[0:2])#1:3
#     return next_cent,data_cent6


# data_cent_5_persist = data_cent5.persist(StorageLevel.MEMORY_ONLY_2)


# # start = timer()
# for i in range(1,k-1):
#     next_selected, data_cent_5_persist = initial_centroids(next_selected,data_cent_5_persist, i)
#     u = [str(i)+'x',str(i)+'y']
#     next_selected = next_selected.toDF(*u)

# # end = timer()
# # print ("Execution time HH:MM:SS:",timedelta(seconds=end-start))



# i= k-1
# data_cent11 = data_cent_5_persist.join(broadcast(next_selected))
# data_cent11 = data_cent11.withColumn(str(i),squaree_spark1(data_cent11.columns[0],data_cent11.columns[1],\
#                                                            data_cent11.columns[k+2],data_cent11.columns[k+3]))
# data_cent11 = data_cent11.drop('mindist').drop(data_cent11.columns[k+2]).drop(data_cent11.columns[k+3])


# def FindMinCOl( *row_list):
#     ind = row_list.index(min(*row_list))
#     return int(ind)

# find_min_val_name = udf(FindMinCOl, IntegerType())
# data_cent14 = data_cent11.withColumn('defined_cluster', find_min_val_name(*data_cent11.columns[2:3+k]))
# data_cent14 = data_cent14.select('first_numeric','second_numeric','defined_cluster')


# # #### Compute weighted average
# # data_cent14 = data_cent14.withColumn('index', row_number().\
# #                                           over(Window.orderBy(monotonically_increasing_id())) - 1)

# # data_cent14 = (data_cent14.join(count, (col('Cindex') == col('index')),\
# #           "inner")).drop('index').drop('Cindex')

# # for c in data_spark.columns:
# #     data_cent14 = data_cent14.withColumn(c+'*',col(c)*col('count') )
# #     data_cent14 = data_cent14.drop(c).withColumnRenamed(c+'*', c)

# # new_centroid = data_cent14.groupBy('defined_cluster').sum('first_numeric', 'second_numeric').\
# # withColumnRenamed('defined_cluster', 'defined_cluster*')

# # new_centroid1 = data_cent14.groupBy('defined_cluster').sum('count')
# # new_centroid = (new_centroid1.join(new_centroid, (col('defined_cluster') == col('defined_cluster*')),\
# #           "inner")).drop('defined_cluster*')
# # ####
# # for c in new_centroid.columns:
# #     new_centroid = new_centroid.withColumnRenamed(c, c.replace('sum(', '').replace(')', ''))

# # for c in new_centroid.columns[2:]:
# #     new_centroid = new_centroid.withColumn(c, col(c) / col('sum(count)'))

# # new_centroid = new_centroid.drop('sum(count)')
# # ####
# new_centroid = data_cent14.groupBy('defined_cluster').avg('first_numeric', 'second_numeric')


# In[ ]:



# spark.sparkContext.getConf().getAll()
# spark.conf.get("spark.sql.shuffle.partitions")
# def UpdateCentroid(x):
#     data_cent_join1 = data_spark.join(broadcast(x))
#     data_cent_join2 = data_cent_join1.withColumn('dist',squaree_spark1(data_cent_join1.columns[0],
#                                                                  data_cent_join1.columns[1],
#                                        data_cent_join1.columns[3],data_cent_join1.columns[4]))#3 4
#     w = Window.partitionBy(data_cent_join2.columns[1])
#     next_centroid = data_cent_join2.withColumn('mindist', F.min('dist').over(w)).\
#     filter(col('dist') == col('mindist')).drop('dist')

# #     #### Compute weighted average
# #     data_cent14 = next_centroid.withColumn('index', row_number().\
# #                                               over(Window.orderBy(monotonically_increasing_id())) - 1)

# #     data_cent14 = (data_cent14.join(count, (col('Cindex') == col('index')),\
# #               "inner")).drop('index').drop('Cindex')

# #     for c in data_spark.columns:
# #         data_cent14 = data_cent14.withColumn(c+'*',col(c)*col('count') )
# #         data_cent14 = data_cent14.drop(c).withColumnRenamed(c+'*', c)

# #     new_centroid = data_cent14.groupBy('defined_cluster').sum('first_numeric', 'second_numeric').\
# #     withColumnRenamed('defined_cluster', 'defined_cluster*')

# #     new_centroid1 = data_cent14.groupBy('defined_cluster').sum('count')
# #     new_centroid = (new_centroid1.join(new_centroid, (col('defined_cluster') == col('defined_cluster*')),\
# #               "inner")).drop('defined_cluster*')


# #     for c in new_centroid.columns[2:]:
# #         new_centroid = new_centroid.withColumn(c, col(c) / col('sum(count)'))

# #     update_new_centroid = new_centroid.drop('sum(count)')
# #     ############
#     update_new_centroid = next_centroid.groupBy('defined_cluster').avg('first_numeric', 'second_numeric')

#     return update_new_centroid, next_centroid

# new_centroid_persist = new_centroid.persist(StorageLevel.MEMORY_AND_DISK)

# # start = timer()

# for i in range(20):
#     new_centroid_persist, final_data = UpdateCentroid(new_centroid_persist)

# # end = timer()


# final_data = final_data.withColumnRenamed('avg(first_numeric)','cent_x').\
# withColumnRenamed('avg(second_numeric)','cent_y')


# In[ ]:


# new_centroid_persist.show()


# In[ ]:


# type(new_centroid_persist)


# In[ ]:


# data_cent11 = sc.parallelize([])
start = timer()

spark_cluster_centroid = sqlContext.createDataFrame(([c.tolist() for c in clusters.centers]),                                                    ['cent_x', 'cent_y'])
spark_cluster_centroid = spark_cluster_centroid.withColumn('defined_cluster', row_number().                                          over(Window.orderBy(monotonically_increasing_id())) - 1)
data_cent11 = 0
for i in range(k):
    u = [ str(i)+'x',str(i)+'y']
    next_selected = spark_cluster_centroid.filter(col('defined_cluster') == str(i)).    drop('defined_cluster').toDF(*u)
    if i == 0:
        data_cent11 = data_spark_df.join(broadcast(next_selected))
#         print(data_cent.count())
        
        data_cent11 = data_cent11.withColumn(str(i),squaree_spark1(data_cent11.columns[0],                            data_cent11.columns[1],data_cent11.columns[i+2],data_cent11.columns[i+3]))
#         data_cent11.show()
        data_cent11 = data_cent11.drop(data_cent11.columns[i+2]).drop(data_cent11.columns[i+3])
#         data_cent11.show()
        data_cent11 = data_cent11.withColumn('mindist',col(str(i)))
#         data_cent11.show()
        data_cent11 = data_cent11.withColumn('mindist1',least(data_cent11.columns[i+2], col('mindist')))
#         data_cent11.show()
        data_cent11 = data_cent11.drop('mindist')
        data_cent11 = data_cent11.withColumnRenamed('mindist1','mindist')
    elif i > 0:
        data_cent11 = data_cent11.join(broadcast(next_selected))
#         data_cent11.show()
        data_cent11 = data_cent11.withColumn(str(i),squaree_spark1(data_cent11.columns[0],                                    data_cent11.columns[1], data_cent11.columns[i+3],data_cent11.columns[i+4]))
#         data_cent11.show()
        data_cent11 = data_cent11.drop(u[0]).drop(u[1])
#         data_cent11.show()
        data_cent11 = data_cent11.withColumn('mindist1',least(data_cent11.columns[i+3], col('mindist')))#4
#         data_cent11.show()
        data_cent11 = data_cent11.drop('mindist')
        data_cent11 = data_cent11.withColumnRenamed('mindist1','mindist')
#         next_cent = data_cent11.orderBy(desc('mindist')).limit(1).select(data_cent11.\
#                                                                          columns[:len(data_spark_df.columns)-1])#1:3
data_cent11 = data_cent11.drop('mindist')

# data_cent11.show()


# In[ ]:


def FindMinCOl( *row_list):
    ind = row_list.index(min(*row_list))
    return int(ind)
find_min_val_name = udf(FindMinCOl, IntegerType())

data_cent11 = data_cent11.withColumn('defined_cluster', find_min_val_name(*data_cent11.columns[2:3+k]))
data_cent11 = data_cent11.select('first','second','defined_cluster')


# In[ ]:


d = data_cent11.toPandas()

end = timer()
print ("Execution time HH:MM:SS:",timedelta(seconds=end-start))


# In[ ]:


# final_data = new_centroid_persist.withColumnRenamed('avg(first_numeric)','cent_x').\
# withColumnRenamed('avg(second_numeric)','cent_y')


# In[ ]:


# final_data = fspark_cluster_centroid.withColumn('index', row_number().\
#                                           over(Window.orderBy(monotonically_increasing_id())) - 1).\
# drop('mindist').drop('second_numeric').drop('first_numeric')


# In[ ]:


# final_data.show()


# In[ ]:


# leaders = leaders.withColumn('index_column_name', row_number().\
#                                           over(Window.orderBy(monotonically_increasing_id())) - 1)
# # final_data.show()


# In[ ]:


# leaders1 = leaders.drop('count')
# leaders1.show()


# In[ ]:


# lead_final = (leaders1
#     .join(final_data, (col('index_column_name') == col('index')), "inner")).drop('1').drop('2').\
#     drop('index_column_name').drop('index')
# #     .withColumnRenamed("defined_cluster", "new label"))
# lead_final.show()


# In[ ]:


# lead_final = lead_final.withColumnRenamed('chunkLabel','chunk')


# In[ ]:


# label_lead_final = (labels
#     .join(lead_final,  ((col('label') == col('old label')) & (col('chunkLabel') == col('chunk'))),\
#           "left")).drop('chunkLabel').drop('label').drop('old label').drop('chunk')

# # label_lead_final.show()


# In[ ]:


# d = label_lead_final.toPandas()
# d


# # Primariy Data: Max( o(n \* p), o(n \* n))
# 

# # means:  uper(n / chunks) \* k \* p

# ## (1 * 1) > 1 / chunks * k * 1 => chunks > k

# # labels: n

# In[ ]:


# data_spark_df.where(col("index_column_name").between(0, chunk-1)).toPandas()


# In[ ]:


# index = list(range(data_spark_df.count()))
# # data_spark_df.withColumn('index', index)


# In[ ]:


# start = timer()

# labelsPred = NPIR(x,k,IR,i)
# # labelsPred1 = NPIR(x1,k,IR,i)
# end = timer()
# print ("Execution time HH:MM:SS:",timedelta(seconds=end-start))
# labelsPred 


# In[ ]:


# xf = pd.DataFrame(x)
# xf['label'] = labelsPred

# xf1 = pd.DataFrame(x1)
# xf1['label'] = labelsPred1


# In[ ]:


# Cs(labelsPred)


# In[ ]:


# leader = []
# f = list(Cs(labelsPred))
# f.sort()
# for i in f:
#     leader.append([np.mean(z) for z in xf[xf['label']==i].values[:,:-1].T])
# leader


# In[ ]:


# leader1 = []
# f = list(Cs(labelsPred1))
# f.sort()
# for i in f:
#     leader1.append([np.mean(z) for z in xf1[xf1['label']==i].values[:,:-2].T])
# leader1


# In[ ]:


# xf1[xf1['label']==0].values[:,:]


# # 1 2 3 4 5 6 7 
# # 1 2 3 4
# # 1 2
# # 1
# 
# # 2**b

# # a \* t(NPIR(n / a)) + (a - 1) \* t(NPIR(k)) ><? t(NPIR(n))

# In[ ]:


# centers = leader.copy()
# centers.extend(leader1)
# centers


# In[ ]:


# newlabel = NPIR(centers,k,IR,i)
# newlabel


# In[ ]:


# newlabel[:k][2]


# In[ ]:


# newleader =[]
# for i in range(k):


# In[ ]:


# xf['newlabel'] = np.full(len(xf), np.nan)
# xf1['newlabel'] =np.full(len(xf1), np.nan)
# # f = list(Cs(labelsPred))
# # f.sort()
# for i in range(k):
#     xf.loc[xf['label'] == i,'newlabel'] = newlabel[:k][i]

# # f = list(Cs(labelsPred1))
# # f.sort()
# for i in range(k):
#     xf1.loc[xf1['label'] == i,'newlabel'] = newlabel[-k:][i]
# merged_data = xf.append(xf1)
# merged_data['label'] = merged_data['newlabel']
# merged_data.drop(['newlabel'], inplace =True,axis=1)
# merged_data.reset_index(inplace=True)
# merged_data.drop(['index'], inplace =True,axis=1)
# merged_data


# In[ ]:


# xf = merged_data.copy()
# labelsPred = merged_data['label']
# xf


# In[ ]:


# tStart = datetime.datetime.now()


# In[ ]:


data = pd.read_csv('blobs2.csv')
data.head()


# In[ ]:


data = data[['0','1','label']]


# In[ ]:


# d = d1.copy()


# In[ ]:


data['new'] = d['defined_cluster'].astype('int8')
# data['new'] = NPIR(pd.read_csv('blobs.csv', header=None).iloc[:,:2].values,k,IR,i)
data.head()


# In[ ]:


y = data['label']
labelsPred = data['new']
# list(labelsPred)


# In[ ]:


x = data[['0','1']].values
x.shape


# In[ ]:


#printing results
print('labels:')
# print(labelsPred)

# tEnd = datetime.datetime.now()
# print('Time: ' + str(tEnd - tStart))
print('Measures:')
print('HS: ' + str(metrics.homogeneity_score(y,labelsPred)))
print('CS: ' + str(metrics.completeness_score(y,labelsPred)))
print('VM: ' + str(metrics.v_measure_score(y,labelsPred)))
print('AMI: ' + str(metrics.adjusted_mutual_info_score(y,labelsPred)))
print('ARI: ' + str(metrics.adjusted_rand_score(y,labelsPred)))


# In[ ]:


import matplotlib.pyplot as plt
from itertools import cycle, islice 

fig = plt.figure()      
colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a','#f781bf', '#a65628', '#984ea3',
                                    '#999999', '#e41a1c', '#dede00']),int(k))))
plt.scatter(x[:, 0], x[:, 1], s=10, color=colors[labelsPred.tolist()])
plt.show()
# plt.clf()


# In[ ]:


# import matplotlib.pyplot as plt
# from itertools import cycle, islice 

# fig = plt.figure()      
# colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a','#f781bf', '#a65628', '#984ea3',
#                                     '#999999', '#e41a1c', '#dede00']),int(k))))
# plt.scatter(x1[:, 0], x1[:, 1], s=10, color=colors[labelsPred1.tolist()])
# plt.show()
# # plt.clf()


# In[ ]:


# import matplotlib.pyplot as plt
# from itertools import cycle, islice 

# fig = plt.figure()      
# colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a','#f781bf', '#a65628', '#984ea3',
#                                     '#999999', '#e41a1c', '#dede00']),int(k))))
# plt.scatter(merged_data.iloc[:, 0].values, merged_data.iloc[:, 1].values, s=10, color=colors[merged_data['label'].astype(int).tolist()])
# plt.show()
# # plt.clf()

