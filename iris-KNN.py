#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import model_selection, neighbors


# In[9]:


df=pd.read_csv("iris.csv")
df.head()


# In[10]:


x=np.array(df.drop(['species'],1))
y=np.array(df['species'])


# In[11]:


y


# In[12]:


x


# In[13]:


x_train,x_test,y_train,y_test=model_selection.train_test_split(x,y,test_size=0.5)
clf=neighbors.KNeighborsClassifier()
clf.fit(x_train,y_train)
accuracy=clf.score(x_test,y_test)
print(accuracy)


# In[14]:


experiment=np.array([[10,120,81,36,1,30,0.299,30],[1,115,70,30,96,34.6,0.529,32],[2,197,70,45,543,30.5,0.158,53]])
experiment=experiment.reshape(3,-1)
predict=clf.predict(experiment)
print(predict)


# In[ ]:





# In[ ]:





# In[ ]:




