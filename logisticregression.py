#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score


# In[12]:


df=pd.read_csv("diabetes.csv")
df


# In[13]:


x=df.drop("Outcome", axis=1)
y=df["Outcome"]


# In[14]:


lda=LinearDiscriminantAnalysis()
x_lda=lda.fit_transform(x,y)


# In[15]:


x_train,x_test,y_train,y_test=model_selection.train_test_split(x_lda,y,test_size=0.3, random_state=0)


# In[16]:


clf=LogisticRegression()
tra=clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)


# In[17]:


y_pred


# In[18]:


print(confusion_matrix(y_test,y_pred))
print(clf.score(x_test,y_test))


# 
