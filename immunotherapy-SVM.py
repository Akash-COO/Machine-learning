#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np
import pandas as pd
from sklearn import model_selection, svm


# In[19]:


df=pd.read_csv('Immunotherapy.csv')


# In[20]:


df


# In[21]:


df.drop(['sex'],1,inplace=True)
df.drop(['age'],1,inplace=True)
df.drop(['Time'],1,inplace=True)


# In[22]:


df


# In[23]:


x=np.array(df.drop(['Result_of_Treatment'],1))
y=np.array(df['Result_of_Treatment'])


# In[24]:


x


# In[25]:


y


# In[45]:


x_train,x_test,y_train,y_test=model_selection.train_test_split(x,y,test_size=0.3)
clf=svm.SVC()
clf.fit(x_train,y_train)
accuracy=clf.score(x_test,y_test)
print(accuracy)


# In[ ]:




