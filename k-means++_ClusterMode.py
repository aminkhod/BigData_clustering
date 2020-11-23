#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import findspark


# In[2]:


findspark.init('/home/rezapour/spark/spark-3.0.0-bin-hadoop2.7')


# In[3]:


import pandas as pd
import numpy as np
from timeit import default_timer as timer
from datetime import timedelta
import matplotlib.pyplot as plt


# In[4]:


from pyspark_dist_explore import hist


# In[5]:


import pyspark


# In[6]:


from pyspark import StorageLevel


# In[7]:


from pyspark.sql import SparkSession


# In[8]:


from pyspark.sql import SparkSession, SQLContext, Window, Row


# In[9]:


from pyspark.sql.types import IntegerType, FloatType, BooleanType, StringType, StructType, StructField,ArrayType
from pyspark.sql.functions import (udf, log, rand, monotonically_increasing_id, col, broadcast,greatest,desc,asc,
row_number, avg, mean, least, struct, lit, sequence)
import pyspark.sql.functions as F


# In[10]:


from pyspark import SparkContext
from pyspark.conf import SparkConf


# In[11]:


from kneed import KneeLocator


# In[12]:



conf = SparkConf()
conf.setMaster('spark://172.23.132.117:7077').setAppName('kemans').set("spark.executor.memory", "18g").set("spark.driver.memory", "10g").set("spark.executor.memoryOverhead", "2000").set("spark.default.parallelism", "2001").set("spark.executor.cores", "7").set("spark.executor.instances", "1")


# In[15]:


sc = SparkContext(conf=conf)


# In[16]:


sc


# In[17]:


sqlContext= SQLContext(sc)


# In[18]:


spark=sqlContext.sparkSession


# In[19]:


spark.conf.set("spark.sql.broadcastTimeout","30000")


# In[ ]:





# In[20]:


def FindMinCOl( *row_list):
    ind = row_list.index(min(*row_list))
    return int(ind)


# In[21]:



find_min_val_name = udf(FindMinCOl, IntegerType())


# In[22]:


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


# In[23]:


squaree_spark1 = udf(lambda x,y,z,r: squaree1(x,y,z,r), FloatType())


# In[24]:


def initial_centroids(next_selected_cent, data_input, i):
    if i == k-1:
        

        data_cent6 = data_input.join(broadcast(next_selected_cent))
       
        data_cent7 = data_cent6.withColumn(str(i),squaree_spark1(data_cent6.columns[0],data_cent6.columns[1], data_cent6.columns[k+2],data_cent6.columns[k+3]))#+3 +4
       
        data_cent8 = data_cent7.drop('mindist').drop(data_cent7.columns[k+2]).drop(data_cent7.columns[k+3])
       
        return data_cent8
        
    else:
        

        data_cent6 = data_input.join(broadcast(next_selected_cent))

        
        data_cent7 = data_cent6.withColumn(str(i), squaree_spark1(data_cent6.columns[0],data_cent6.columns[1],data_cent6.columns[i+3], data_cent6.columns[i+4]))

        
        data_cent8 = data_cent7.drop(data_cent7.columns[i+3]).            drop(data_cent7.columns[i+4])
        
       
            
        data_cent9 = data_cent8.withColumn('mindist1',least(data_cent8.columns[i+3], col('mindist')))
       

        data_cent10 = data_cent9.drop('mindist')

        
        data_cent12 = data_cent10.withColumnRenamed('mindist1', 'mindist')
        
        data_cent13 = data_cent12.repartition(2001)
       
      
        next_cent_cache = data_cent13.orderBy(desc('mindist')).limit(1).cache()
        
        next_cent = next_cent_cache.select(data_cent12.columns[0:2])
        
        return next_cent, data_cent12


# In[25]:


def UpdateCentroid(first_read_data, centroids,num_clusters):
    data_cent_join1 = first_read_data.join(broadcast(centroids))
    

    data_cent_join2 = data_cent_join1.withColumn('dist',
            squaree_spark1(data_cent_join1.columns[0], data_cent_join1.columns[1],
            data_cent_join1.columns[3],data_cent_join1.columns[4]))  


    w = Window.partitionBy(data_cent_join2.columns[1], data_cent_join2.columns[0])

    next_cent = data_cent_join2.withColumn('mindist', F.min('dist').over(w)).        filter(col('dist') == col('mindist')).drop('dist')
    
 
    next_cent1 = next_cent.repartition(num_clusters,'defined_cluster')
    
    update_new_centroid = next_cent1.groupBy('defined_cluster').avg('first_numeric', 'second_numeric')
    
   

    return update_new_centroid, next_cent


# In[26]:


def cost_function(label_data, final_label):
    start = timer()
    
    true_label = spark.read.format('csv').option('header','True').option('index','True').load(label_data)

    label_list = []
    for i in range(k):
        label_list.append(true_label.filter(col('0') == i).count())
    number = len(final_label)
    label_array = np.array(label_list)
    my_label = []
    for i in range(k):
        my_label.append(len(np.where(final_label == i)[0]))
    real_label = []
    for i in range(k):
        real_label.append(len(np.where(label_list == i)[0]))
    error = np.sum(np.absolute([my_label - label_list for my_label, label_list in zip(my_label, label_list)]))
    accuracy = ((number - error) / number) * 100
    print(accuracy)
    end = timer()
    print ("Execution time HH:MM:SS:",timedelta(seconds=end-start))


# In[27]:


def num_cluster(first_range, last_range):
    
    total_dist = []
    last_range = last_range + 1

    t = 101
    for i in range(first_range,last_range):
        t += 1
        
        dataset = 'test_data_{}'.format(t+2)+'.csv'
        label_data = 'y_{}'.format(t+2)+'.csv'
        output_df = my_kmeans(dataset, i, label_data)
        total_dist.append(output_df.agg(F.sum("mindist")).collect()[0][0])
               
    
    kl = KneeLocator(
        range(last_range, last_range), total_dist, curve="convex", direction="decreasing"
    )
   
    
    plt.xlabel('number of clusters k')
    plt.ylabel('Sum of squared distances')
    plt.plot(range(first_range,last_range), total_dist, 'bx-')
    plt.vlines(kl.knee, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
    
    return total_dist


# In[28]:


def my_kmeans(dataset, k, output_label ):
    
    data_spark_df01 = spark.read.format('csv').option('header','True').option('index','True').load(dataset)
    new_name = ['first', 'second']
    data_spark_df0 = data_spark_df01.toDF(*new_name)
    
    
    
    data_spark_df1 = data_spark_df0.withColumn("first_numeric", data_spark_df0["first"].cast(FloatType()))
    data_spark_df2 = data_spark_df1.withColumn("second_numeric", data_spark_df1["second"].cast(FloatType()))
    data_spark_df = data_spark_df2.drop('first').drop('second')

    df_centroid = data_spark_df.sample(False, 0.1,seed = 0)
    df_centroid_cache = df_centroid.limit(1).cache()
    df_centroid_cache.show()
    
    new_name = ['x','y']

    df_centroid_cache = df_centroid_cache.toDF(*new_name)
    
    i = 0

    data_cent = data_spark_df.join(broadcast(df_centroid_cache))
    data_cent = data_cent.withColumn(str(i),squaree_spark1(data_cent.columns[0],data_cent.columns[1],
                                                  data_cent.columns[2*i+2],data_cent.columns[2*i+3]))


    data_cent = data_cent.drop(data_cent.columns[i+2]).drop(data_cent.columns[i+3])


    data_cent3 = data_cent.withColumn('mindist',col(str(i)))

    data_cent4 = data_cent3.withColumn('mindist1',least(data_cent3.columns[i+2], col('mindist'))).drop('mindist')

    


    data_cent5 = data_cent4.withColumnRenamed('mindist1','mindist')


    next_selected_cache = data_cent5.orderBy(desc('mindist')).limit(1).cache()

    next_selected = next_selected_cache.select(data_cent5.columns[0:2])

    u = [str(i)+'x',str(i)+'y']
    next_selected = next_selected.toDF(*u)
    data_cent5.explain()
    
    
    start = timer()
    for i in range(1,k,1):
        print(i)

        next_selected_take = next_selected.repartition(2001).cache()#
        next_selected_take.take(1)

        if i == k-1:
            global data_cent11

            data_cent11 = initial_centroids(next_selected_take,data_cent5, i)

        else:
            next_selected_take, data_cent5 = initial_centroids(next_selected_take,data_cent5, i)

            u = [str(i)+'x',str(i)+'y']

            next_selected_take = next_selected_take.toDF(*u)

            next_selected = next_selected_take



    end = timer()
    print ("Execution time HH:MM:SS:",timedelta(seconds=end-start))
    
    data_cent14 = data_cent11.withColumn('defined_cluster',find_min_val_name(*data_cent11.columns[2:3+k])) 

    data_cent16 = data_cent14.select('first_numeric','second_numeric','defined_cluster')

    next_cent17 = data_cent16.repartition(k,'defined_cluster')
    new_centroid = next_cent17.groupBy('defined_cluster').avg('first_numeric', 'second_numeric')
    
    start = timer()

    for i in range(20):
        print(i)
        new_centroid_cache_take = new_centroid.repartition(2001).cache()
        new_centroid_cache_take.take(1)

        new_centroid_cache_take, final_data = UpdateCentroid(data_spark_df,new_centroid_cache_take, k)


        new_centroid = new_centroid_cache_take


    end = timer()
    print ("Execution time HH:MM:SS:",timedelta(seconds=end-start))
    
    final_data1 = final_data.select('defined_cluster')
    final_list = final_data1.toPandas()
    final_label = np.array(list(final_list['defined_cluster']))
    
    #cost_function(output_label, final_label)
   
    
    return  final_data

    


# In[29]:


#if number of clusters is not defined
#num_cluster(first_range, last_range)


# In[ ]:


#if number of clusters is defined
dataset = 'test_data_103.csv'
k = 4
label_data = 'y_103.csv'


df_result = my_kmeans(dataset, k, label_data)


# In[31]:


df_result.write.csv('output.csv')

