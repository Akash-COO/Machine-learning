#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from sklearn import model_selection, svm
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


df=pd.read_csv("Nheart.csv")


# In[4]:


df.replace('?',-9999,inplace=True)
df.drop(['age'],1,inplace=True)


# In[5]:


df.head()


# In[6]:


df.drop(['sex'],1,inplace=True)


# In[7]:


df.head()


# In[8]:


x=np.array(df.drop(['target'],1))
y=np.array(df['target'])
x


# In[9]:


y


# In[10]:


x_train,x_test,y_train,y_test=model_selection.train_test_split(x,y,test_size=0.2)
clf=svm.SVC()
clf.fit(x_train,y_train)
accuracy=clf.score(x_test,y_test)
print(accuracy)


# In[106]:


#a=pd.read_csv("Nheart.csv")
#sns.FacetGrid(a,hue='target',size=4).map(plt.scatter,'age','chol').add_legend()
#plt.show()


# In[ ]:




