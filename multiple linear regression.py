#!/usr/bin/env python
# coding: utf-8

# In[3]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import linear_model


# In[4]:


df=pd.read_csv("diabetes.csv")
df.head()


# In[5]:


x=df.drop(['Outcome'],1)
y=df['Outcome']


# In[6]:


print(x)
print(y)


# In[7]:


from sklearn import datasets


# In[13]:


diabetes=datasets.load_diabetes()
#data is stored in the form of dictionary
#print(diabetes)


# In[9]:


print(diabetes.keys())


# In[25]:


diabetes_X=diabetes.data
print(diabetes_X)


# In[26]:


diabetes_X_train=diabetes_X[:-30]
diabetes_X_test=diabetes_X[-20:]


# In[27]:


diabetes_Y_train=diabetes.target[:-30]
diabetes_Y_test=diabetes.target[-20:]


# In[28]:


#simple linear regression
slr=linear_model.LinearRegression()
slr.fit(diabetes_X_train,diabetes_Y_train)
diabetes_Y_predict=slr.predict(diabetes_X_test)


# In[29]:


from sklearn.metrics import mean_squared_error
print("MEAN SQUARED ERROR: ",mean_squared_error(diabetes_Y_test,diabetes_Y_predict))


# In[30]:


print("WEIGHT : ",slr.coef_)
print("INTERCEPT : ",slr.intercept_)


# In[ ]:




