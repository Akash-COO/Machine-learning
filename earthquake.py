#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv('Major EarthQuakes in Greece.csv')


# In[10]:


df.rename(columns={'Date':'Day', 'LATATITUDE (N)':'Lat', 'LONGITUDE  (E)' : 'Long', 'MAGNITUDE (Richter)' : 'Magn' }, inplace=True)


# In[11]:


df.info()
df.describe()


# In[4]:


quake_x=df.drop(['''MAGNITUDE (Richter)'''],1)
quake_y=df['''MAGNITUDE (Richter)''']


# In[5]:


quake_x


# In[6]:


sns.heatmap(quake_x.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[7]:


sns.countplot(x='Month',data=df,palette='rainbow')


# In[8]:


sns.distplot(quake_y.dropna(),kde=False,color='red',bins=30)


# In[9]:


sns.countplot(x='Year',data=df,palette='rainbow')


# In[ ]:


groupby('')['TargetValue'].mean().plot(kind = 'bar', figsize= (40,20), title= "Countries with COVID-19 MAX", color='red')

