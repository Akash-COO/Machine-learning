#!/usr/bin/env python
# coding: utf-8

# In[40]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.cluster import KMeans
from sklearn import preprocessing, model_selection
from sklearn.linear_model import LogisticRegression#binaryclassifier
from sklearn import model_selection, neighbors, svm


# In[2]:


df=pd.read_csv('brain_cancer.csv')


# In[3]:


df


# In[4]:


df.drop(['samples'],1,inplace=True)


# In[5]:


df


# In[6]:


x=np.array(df.drop(['type'],1))
y=np.array(df['type'])


# In[21]:


x


# In[8]:


y


# In[22]:


x_train,x_test,y_train,y_test=model_selection.train_test_split(x,y,test_size=0.5)


# In[51]:


clf1=neighbors.KNeighborsClassifier()
clf1.fit(x_train,y_train)
acc1=clf1.score(x_test,y_test)
print(acc1)


# In[30]:


clf2=svm.SVC()
clf2.fit(x_train,y_train)
acc2=clf2.score(x_test,y_test)
print(acc2)


# In[33]:


dftree=DecisionTreeClassifier()
tra=dftree.fit(x,y)
acc3=dftree.score(x,y)
print(acc3)


# In[35]:


#from sklearn.externals.six import StringIO
#from IPython.display import Image
#from sklearn.tree import export_graphviz
#import pydotplus
#dot_data=StringIO()
#export_graphviz(dftree, out_file=dot_data,filled=True, rounded=True, special_characters=True)
#graph=pydotplus.graph_from_dot_data(dot_data.getvalue())
#Image(graph.create_png())


# In[38]:


rnd_clf=RandomForestClassifier(n_estimators=500)
log_clf=LogisticRegression()
voting_clf=VotingClassifier(estimators=[('rf',rnd_clf),('log_reg',log_clf)],voting='hard')
voting_clf.fit(x,y)


# In[39]:


from sklearn.metrics import accuracy_score
for clf in (rnd_clf, log_clf, voting_clf):
    clf.fit(x,y)
    y_pred=clf.predict(x)
    print(clf.__class__.__name__, accuracy_score(y,y_pred))


# In[41]:


df.infer_objects()
#to convert the character objects into numeric form.
df.fillna(0, inplace=True)
#fillna is used to pick NaN value from data and replace it with given value.
df.head(25)


# In[42]:


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


# In[43]:


x=preprocessing.scale(x)


# In[44]:


x


# In[50]:


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


# In[ ]:


x=[1,2,3,4,5]
popularity=[80,74,86,40,95]
anime=['shinchan','doremon','tom & jerry','ninja hattori','Mr. Bean']
mycolor=['b','g','r','b','m']
plt.bar(x,popularity,tick_label=anime,width=0.5,color=mycolor)
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title('accuracy bar graph')
plt.show()
plt.close()

