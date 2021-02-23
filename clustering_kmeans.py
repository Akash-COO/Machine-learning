#!/usr/bin/env python
# coding: utf-8

# In[1]:


#clustering is an un-supervised learning. 
#of two types flat and hirarichal
#algo take data in the form of array


# In[36]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.cluster import KMeans
from sklearn import preprocessing, model_selection


# In[30]:


style.use('ggplot')
df=pd.read_excel('titanic.xls')
df.head(25)


# In[31]:


df.drop(['name'],1,inplace=True)


# In[32]:


df.infer_objects()
#to convert the character objects into numeric form.
df.fillna(0, inplace=True)
#fillna is used to pick NaN value from data and replace it with given value.
df.head(25)


# In[33]:


def handle_non_numerical_data(df):
    columns=df.columns.values#creating a list name column that contain all column headings
    for column in columns:
        text_digit_vals={}#picks values of all columns individually
        def convert_to_int(val):
            return text_digit_vals[val]
        if df[column].dtype!=np.int64 and df[column].dtype!=np.float64:
            column_contents=df[column].values.tolist()#saving data of individual in a list
            unique_elements=set(column_contents)#removing duplicacy
            x=0
            for unique in unique_elements:
                if unique not in text_digit_vals:#check whether tha value is already present or not if not value of x is assigned to it
                    text_digit_vals[unique]=x
                    x+=1
            df[column]=list(map(convert_to_int, df[column]))#df[column] contain unordered data and map(convert_to_int) is used to map assigned value to the key value
    return df
df=handle_non_numerical_data(df)
df.head()
df


# In[62]:


df.drop(['boat'],1,inplace=True)
x=np.array(df.drop(['survived'],1).astype(float))
x=preprocessing.scale(x)#transformation of data-convert data into binary form
y=np.array(df['survived'])


# In[63]:


x


# In[66]:


clf=KMeans(n_clusters=2)
clf.fit(x)
correct=0
for i in range(len(x)):
    predict_me=np.array(x[i].astype(float))
    predict_me=predict_me.reshape(-1,len(predict_me))
    prediction=clf.predict(predict_me)
    if prediction[0]==y[i]:
        correct+=1
print(correct/len(x))


# In[65]:


a=pd.read_excel("titanic.xls")


# In[47]:


print(a['survived'].value_counts())#to count the value of a column


# In[67]:


mycolors=['r','g']
plt.pie([809,500],labels=['died','survived'],colors=mycolors,startangle=90,shadow=True,
       explode=(0.1,0.1,),radius=2.0,autopct='%1.2f%%')


# In[53]:


a['survived'].value_counts().plot(kind='bar')
plt.show()

