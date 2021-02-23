#!/usr/bin/env python
# coding: utf-8

# In[18]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage


# In[21]:


#x=np.array([[5,3],[10,5],[15,12],[24,10],[30,45],[85,70],[71,80],[60,78],[55,52],[80,91]])
df=pd.read_csv('iris.csv')
df


# In[30]:


x=df.iloc[:,[1,2]]
print(x)
x1=x["sepal_width"]
x2=x["petal_length"]
colors=['','','']
plt.scatter(x1,x2)


# In[32]:


linked=linkage(x,'single')
print(linked)
labelList=range(1,151)


# In[34]:


dendrogram(linked,
         orientation='top',
         labels=labelList,
         distance_sort='acending',
         show_leaf_counts=True)
plt.show()


# In[14]:


from sklearn.cluster import AgglomerativeClustering


# In[35]:


cluster=AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
cluster.fit_predict(x)
print(cluster.labels_)
plt.scatter(x1,x2)
plt.show()


# In[ ]:




