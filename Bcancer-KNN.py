#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
plot1=[1,3]
plot2=[2,5]
euclidean_distance=math.sqrt( (plot1[0]-plot2[0])**2+(plot1[1]-plot2[1])**2)
euclidean_distance


# In[2]:


import numpy as np
import seaborn as sns
from sklearn import model_selection, neighbors
import pandas as pd
import matplotlib.pyplot as plt


# In[5]:


df=pd.read_csv('breast-cancer-wisconsin.txt')
df.replace('?',-9999,inplace=True)
df.drop(['id'],1,inplace=True)
df


# In[3]:


x=np.array(df.drop(['class'],1))
y=np.array(df['class'])
x


# In[4]:


y


# In[12]:


sns.FacetGrid(df,hue='class',size=4).map(plt.scatter,'clump_thickness','class').add_legend()
plt.show()


# In[15]:


x_train,x_test,y_train,y_test=model_selection.train_test_split(x,y,test_size=0.2,random_state=0)
clf=neighbors.KNeighborsClassifier()
clf.fit(x_train,y_train)
accuracy=clf.score(x_test,y_test)
print(accuracy)


# In[16]:


example_measures=np.array([[1,2,1,2,1,2,1,2,2],[4,5,6,7,8,9,8,7,9]])
example_measures=example_measures.reshape(2,-1)
prediction=clf.predict(example_measures)
print(prediction)


# In[ ]:




