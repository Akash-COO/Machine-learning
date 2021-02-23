#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression#binaryclassifier


# In[4]:


df=pd.read_csv('iris.csv')


# In[5]:


df


# In[6]:


df.drop(['Id'],1,inplace=True)


# In[7]:


df.replace('?',1,inplace=True)


# In[8]:


df


# In[9]:


x=np.array(df.drop(['Species'],1))
y=np.array(df['Species'])


# In[10]:


x


# In[11]:


y


# In[12]:


rnd_clf=RandomForestClassifier(n_estimators=500)
tree_clf=DecisionTreeClassifier(max_depth=2)
svm_clf=SVC()
knn_clf=KNeighborsClassifier()
log_clf=LogisticRegression()
voting_clf=VotingClassifier(estimators=[('rf',rnd_clf),('svc',svm_clf),('tree',tree_clf),('log_reg',log_clf),('knn',knn_clf)],voting='hard')
voting_clf.fit(x,y)


# In[14]:


from sklearn.metrics import accuracy_score
for clf in (rnd_clf, svm_clf, tree_clf, log_clf, log_clf, knn_clf, voting_clf):
    clf.fit(x,y)
    y_pred=clf.predict(x)
    print(clf.__class__.__name__, accuracy_score(y,y_pred))

