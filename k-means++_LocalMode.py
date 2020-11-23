
# coding: utf-8

# In[1]:


import findspark


# In[2]:


import pandas as pd
import numpy as np
from timeit import default_timer as timer
from datetime import timedelta
import matplotlib.pyplot as plt


# In[3]:


findspark.init('/home/fatemeh/spark-3.0.0-preview-bin-hadoop2.7')


# In[4]:


import pyspark


# In[5]:


from pyspark import StorageLevel


# In[6]:


from pyspark.sql.types import IntegerType, FloatType, BooleanType, StringType, StructType, StructField,ArrayType
from pyspark.sql.functions import udf, log, rand, monotonically_increasing_id, col, broadcast,greatest,desc,asc, row_number, avg, mean, least, struct, lit, sequence
import pyspark.sql.functions as F


# In[7]:


from pyspark.sql import SparkSession, SQLContext, Window, Row
from pyspark import SparkConf


# In[8]:


spark = SparkSession.builder.master("local[*]").config("spark.sql.broadcastTimeout", "30000s").config("spark.network.timeout","30000s").config("spark.executor.heartbeatInterval","12000000ms").config("spark.storage.blockManagerSlaveTimeoutMs","12000001ms").config("spark.driver.maxResultSize","5g").config("spark.default.parallelism", "100").config("spark.memory.offHeap.enabled","true").config("spark.memory.offHeap.size", "16g").appName("mykmeans").getOrCreate()


# In[9]:


sc = spark.sparkContext


# In[10]:


sqlContext = SQLContext(sc)


# In[11]:


#read csv
data_spark_df = spark.read.format('csv').option('header','True').option('index','True').load('test_data_102.csv')


# In[12]:


data_spark_df


# In[50]:


data_spark_df.describe()


# In[51]:


data_spark_df.printSchema()


# In[12]:


data_spark_df.show()


# In[13]:


new_name = ['first', 'second']
data_spark_df = data_spark_df.toDF(*new_name)


# In[14]:


data_spark_df.show()


# In[15]:


#data_spark_df = data_spark_df.repartition(1000)


# In[14]:


spark.conf.set("spark.sql.debug.maxToStringFields", 1000)


# In[15]:


spark.conf.set('spark.jars.packages','com.databricks:spark-cav_2.11')


# In[16]:


spark.conf.set("spark.sql.parquet.compression.codec","gzip")


# In[17]:


spark.conf.set("spark.sql.execution.arrow.enabled", "true")


# In[18]:


sqlContext.setConf("spark.sql.shuffle.partitions", "10")


# In[19]:


def MyCheckUpdate(a, b, c, d):
    a = float(a)
    b = float(b)
    c = float(c)
    d = float(d)
    res = (a-c) + (b-d)
    if res == 0:
        
        return 1.0
    return 0.0


# In[20]:


check_centroid = udf(lambda x,y,z,r: MyCheckUpdate(x,y,z,r), FloatType())


# In[21]:


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


# In[22]:


squaree_spark1 = udf(lambda x,y,z,r: squaree1(x,y,z,r), FloatType())


# In[23]:


sqlContext.sql("SET spark.sql.autoBroadcastJoinThreshold = -1")


# In[24]:


data_spark_df = data_spark_df.withColumn("first_numeric", data_spark_df["first"].cast(FloatType()))
data_spark_df = data_spark_df.withColumn("second_numeric", data_spark_df["second"].cast(FloatType()))
data_spark_df = data_spark_df.drop('first').drop('second')


# In[25]:


df_centroid = data_spark_df.sample(False, 0.4,seed = 0).limit(1).cache()

new_name = ['x','y']
df_centroid = df_centroid.toDF(*new_name)


# In[28]:


df_centroid.show()


# In[26]:


df_centroid.show()


# In[27]:


#just for first round
#first time
i = 0

data_cent = data_spark_df.join(broadcast(df_centroid))

data_cent1 = data_cent.withColumn(str(i),squaree_spark1(data_cent.columns[0],data_cent.columns[1],
                                              data_cent.columns[2*i+2],data_cent.columns[2*i+3]))

data_cent2 = data_cent1.drop(data_cent1.columns[i+2]).drop(data_cent1.columns[i+3])

data_cent3 = data_cent2.withColumn('mindist',col(str(i)))

data_cent4 = data_cent3.withColumn('mindist1',least(data_cent3.columns[i+2], col('mindist')))

data_cent4 = data_cent4.drop('mindist')

data_cent5 = data_cent4.withColumnRenamed('mindist1','mindist')

next_selected = data_cent5.orderBy(desc('mindist')).limit(1).select(data_cent5.columns[0:2])#1:3


df_centroid = df_centroid.union(next_selected)

u = [str(i)+'x',str(i)+'y']
next_selected = next_selected.toDF(*u)


# In[28]:


def initial_centroids(next_selected,data_cent_5_persist, i):

   data_cent6 = data_cent_5_persist.join(broadcast(next_selected))


   data_cent6 = data_cent6.withColumn(str(i),squaree_spark1(data_cent6.columns[0],data_cent6.columns[1],
                                             data_cent6.columns[i+3],data_cent6.columns[i+4]))#+4 +5
   
   
   data_cent6 = data_cent6.drop(data_cent6.columns[i+3]).drop(data_cent6.columns[i+4])#+4 +5


   data_cent6 = data_cent6.withColumn('mindist1',least(data_cent6.columns[i+3], col('mindist')))#4

   data_cent6 = data_cent6.drop('mindist')
   
   data_cent6 = data_cent6.withColumnRenamed('mindist1','mindist')
   
   next_cent = data_cent6.orderBy(desc('mindist')).limit(1).select(data_cent6.columns[0:2])#1:3


   return next_cent,data_cent6


# In[29]:


data_cent_5_persist = data_cent5.persist(StorageLevel.MEMORY_ONLY_2)


# In[30]:


k=4


# In[31]:


start = timer()
for i in range(1,k-1,1):
    
    next_selected, data_cent_5_persist = initial_centroids(next_selected,data_cent_5_persist, i)


    u = [str(i)+'x',str(i)+'y']

    next_selected = next_selected.toDF(*u)



end = timer()
print ("Execution time HH:MM:SS:",timedelta(seconds=end-start))


# In[32]:


#df_centroid.show()


# In[33]:


i= k-1

data_cent11 = data_cent_5_persist.join(broadcast(next_selected))
data_cent11 = data_cent11.withColumn(str(i),squaree_spark1(data_cent11.columns[0],data_cent11.columns[1],                                                       data_cent11.columns[k+2],data_cent11.columns[k+3]))
data_cent11 = data_cent11.drop('mindist').drop(data_cent11.columns[k+2]).drop(data_cent11.columns[k+3])


# In[34]:


def FindMinCOl( *row_list):
    ind = row_list.index(min(*row_list))
    return int(ind)


# In[35]:


find_min_val_name = udf(FindMinCOl, IntegerType())


# In[36]:


data_cent14 = data_cent11.withColumn('defined_cluster',find_min_val_name(*data_cent11.columns[2:3+k]))


# In[37]:


data_cent14 = data_cent14.select('first_numeric','second_numeric','defined_cluster')


# In[41]:


data_cent14.filter(col("defined_cluster") == 2).show()


# In[51]:


data_cent14.show()


# In[38]:


new_centroid = data_cent14.groupBy('defined_cluster').avg('first_numeric', 'second_numeric')


# In[54]:


new_centroid.show()


# In[39]:


spark.sparkContext.getConf().getAll()


# In[47]:


spark.conf.get("spark.sql.shuffle.partitions") 


# In[42]:


next_centroid.show()


# In[44]:


next_centroid1.show()


# In[39]:



def UpdateCentroid(x):
 
 
 data_cent_join1 = data_spark_df.join(broadcast(x))
 
 data_cent_join2 = data_cent_join1.withColumn('dist',squaree_spark1(data_cent_join1.columns[0],
                                                                 data_cent_join1.columns[1],
                                       data_cent_join1.columns[3],data_cent_join1.columns[4]))#3 4
 w = Window.partitionBy(data_cent_join2.columns[1])

 next_centroid = data_cent_join2.withColumn('mindist', F.min('dist').over(w)).filter(col('dist') == col('mindist'))     .drop('dist')
 

 update_new_centroid = next_centroid.groupBy('defined_cluster').avg('first_numeric', 'second_numeric')

 
 return update_new_centroid, next_centroid
 


# In[40]:


new_centroid_persist = new_centroid.persist(StorageLevel.MEMORY_AND_DISK)


# In[41]:



start = timer()

for i in range(20):
    
    new_centroid_persist, final_data = UpdateCentroid(new_centroid_persist)

end = timer()
print ("Execution time HH:MM:SS:",timedelta(seconds=end-start))


# In[42]:


final_data = final_data.withColumnRenamed('avg(first_numeric)','cent_x').withColumnRenamed('avg(second_numeric)','cent_y')


# In[43]:


final_data1 = final_data.select('defined_cluster')


# In[44]:


final_data1.count()


# In[47]:


final_data3 = final_data1.toPandas()


# In[48]:


#read csv
true_label = spark.read.format('csv').option('header','True').option('index','True').load('y_102.csv')


# In[49]:


true_label_list = final_data1.toPandas()


# In[51]:


true_label_list


# In[43]:


final_data1.write.csv('cluster_label_718.csv')


# In[ ]:


final_data.repartition(1).write.csv('kmeans_output11.csv')

