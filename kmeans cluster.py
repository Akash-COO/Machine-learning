#!/usr/bin/env python
# coding: utf-8

# In[3]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
import pandas as pd
from sklearn.cluster import KMeans


# In[5]:


iris=datasets.load_iris()


# In[6]:


x=iris.data[:,:2]
y=iris.target


# In[7]:


plt.scatter(x[:,0],x[:,1],c=y,cmap='rainbow')
plt.xlabel('sepal_length')
plt.ylabel('sepal_width')


# In[8]:


km=KMeans(n_clusters=3)
km.fit(x)


# In[11]:


print(km.cluster_centers_)
new_label=km.labels_
fig,axes=plt.subplots(1,2,figsize=(16,8))
axes[0].scatter(x[:,0],x[:,1],c=new_label,cmap='rainbow')
axes[0].set_title('KMeans_points')

axes[1].scatter(x[:,0],x[:,1],c=y,cmap='rainbow')
axes[1].set_title('normal points')


# In[ ]:




