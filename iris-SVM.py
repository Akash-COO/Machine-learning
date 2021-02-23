#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn import model_selection, svm
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv('iris.csv')


# In[2]:


df


# In[3]:


df.replace('?',-9999,inplace=True)
df.drop(['Id'],1,inplace=True)
df.head()


# In[7]:


x=np.array(df.drop(['Species'],1))
y=np.array(df['Species'])


# In[8]:


x


# In[9]:


y


# In[14]:


x_train,x_test,y_train,y_test=model_selection.train_test_split(x,y,test_size=0.3)
clf=svm.SVC()
clf.fit(x_train,y_train)
accuracy=clf.score(x_test,y_test)
print(accuracy)


# In[23]:


experiment=np.array([[5.1,3.5,1.4,0.2],[5.8,2.7,5.1,1.0],[5.8,2.8,5.1,2.4]])
experiment=experiment.reshape(3,-1)
predict=clf.predict(experiment)
print(predict)

