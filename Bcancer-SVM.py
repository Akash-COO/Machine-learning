#!/usr/bin/env python
# coding: utf-8

# In[1]:


#svm satnds for support vector machine
import numpy as np
from sklearn import model_selection, svm
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv('breast-cancer-wisconsin.txt')
df.replace('?',-9999,inplace=True)
df.drop(['id'],1,inplace=True)
df.head()


# In[2]:


x=np.array(df.drop(['class'],1))
y=np.array(df['class'])
x


# In[3]:


y


# In[4]:


x_train,x_test,y_train,y_test=model_selection.train_test_split(x,y,test_size=0.2)
clf=svm.SVC()
clf.fit(x_train,y_train)
accuracy=clf.score(x_test,y_test)
print(accuracy)


# In[12]:


experiment=np.array([[1,2,3,2,3,1,2,3,4],[6,7,8,9,9,8,7,6,9]])
experiment=experiment.reshape(2,-1)
predict=clf.predict(experiment)
print(predict)

