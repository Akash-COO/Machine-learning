#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np
import pandas as pd
from sklearn import model_selection,neighbors


# In[12]:


df=pd.read_csv('Immunotherapy.csv')


# In[13]:


df


# In[14]:


df.drop(['sex'],1,inplace=True)
df.drop(['Time'],1,inplace=True)


# In[15]:


df.drop(['age'],1,inplace=True)


# In[16]:


df.replace('?',1,inplace=True)


# In[17]:


df


# In[19]:


x=np.array(df.drop(['Result_of_Treatment'],1))
y=np.array(df['Result_of_Treatment'])


# In[20]:


x


# In[21]:


y


# In[39]:


x_train,x_test,y_train,y_test=model_selection.train_test_split(x,y,test_size=0.3)
clf=neighbors.KNeighborsClassifier()
clf.fit(x_train,y_train)
accuracy=clf.score(x_test,y_test)
print(accuracy)


# In[ ]:




