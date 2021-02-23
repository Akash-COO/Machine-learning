#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np
from sklearn.cluster import MeanShift
from sklearn.datasets.samples_generator import make_blobs
#blobs is a function used to create randomized data in the form of numbers
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import style


# In[15]:


style.use('ggplot')
centers=[[1,1,1],[5,5,5],[3,10,10]]
x,_=make_blobs(n_samples=100,centers=centers,cluster_std=1.5)
#'_' is also a variable


# In[16]:


ms=MeanShift()
ms.fit(x)
labels=ms.labels_
cluster_centers=ms.cluster_centers_
print(cluster_centers)


# In[17]:


n_clusters_=len(np.unique(labels))
print("No. of estimated clusters : ",n_clusters_)


# In[19]:


colors=10*['r','g','b','c','k','y','m']
fig=plt.figure()
ax=fig.add_subplot(111, projection='3d')
for i in range(len(x)):
    ax.scatter(x[i][0],x[i][1],x[i][2],c=colors[labels[i]], marker='o')
ax.scatter(cluster_centers[:,0], cluster_centers[:,1], cluster_centers[:,2],
marker="x", color='k', s=150, linewidth=5, zorder=10)
plt.show()


# In[7]:


from sklearn.cluster import MeanShift
import numpy as np
x=np.array([[1,1],[2,1],[1,1],
           [4,7],[3,5],[3,6]])
clustering=MeanShift().fit(x)
clustering.labels_
#labels_ is used whenever a data point comes to create a centroid we need to give it a label.
#labels_ function perform this task


# In[10]:


print(clustering.predict([[0,0],[5,5]]))


# In[11]:


print(clustering)


# In[ ]:


#n_jobs can be either 1 or -1 multi-threading
#preprocessing is used  where the data is heavy

