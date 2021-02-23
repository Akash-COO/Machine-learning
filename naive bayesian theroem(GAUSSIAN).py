#!/usr/bin/env python
# coding: utf-8

# In[25]:


import pandas as pd
import numpy as np
from sklearn import model_selection, naive_bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score


# In[15]:


df=pd.read_csv("iris.csv")
df


# In[10]:


x=df.drop("species", axis=1)
y=df["species"]


# In[12]:


x


# In[13]:


x


# In[73]:


x_train,x_test,y_train,y_test=model_selection.train_test_split(x,y,test_size=0.3, random_state=0)


# In[74]:


gnb=GaussianNB()
gnb1=gnb.fit(x_train,y_train)
y_pred=gnb1.predict(x_test)


# In[75]:


y_pred


# In[76]:


print(confusion_matrix(y_test,y_pred))
accuracy=gnb.score(x_test,y_test)


# In[77]:


accuracy


# In[ ]:





# In[ ]:




