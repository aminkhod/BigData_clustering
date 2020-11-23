
# coding: utf-8

# In[1]:


import findspark
findspark.init()
from pyspark import SparkContext
sc = SparkContext("local", "First App")


# In[10]:


logFile = "/home/amin/Github/BigData_clustering/Untitled.ipynb"  

logData = sc.textFile(logFile).cache()
numAs = logData.filter(lambda s: 'a' in s).count()
numBs = logData.filter(lambda s: 'b' in s).count()
print("Lines with a: %i, lines with b: %i" % (numAs, numBs))


# In[23]:


words = sc.parallelize (
   ["scala", 
   "java", 
   "hadoop", 
   "spark", 
   "akka",
   "spark vs hadoop", 
   "pyspark",
   "pyspark and spark"]
)
words


# In[24]:


counts = words.count()
print("Number of elements in RDD -> %i" % (counts))


# In[25]:


coll = words.collect()
print("Elements in RDD -> %s" % (coll))


# In[27]:


def f(x):
    print(x)
fore = words.foreach(f)


# In[31]:


words.foreach(f)

