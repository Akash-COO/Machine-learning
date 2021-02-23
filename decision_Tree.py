#!/usr/bin/env python
# coding: utf-8

# In[20]:


import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score


# In[26]:


df=pd.read_csv("iris.csv")
df


# In[27]:


x=df.drop("species", axis=1)
y=df["species"]


# In[36]:


lda=LinearDiscriminantAnalysis()
x_lda=lda.fit_transform(x,y)


# In[37]:


x_train,x_test,y_train,y_test=model_selection.train_test_split(x_lda,y,test_size=0.3, random_state=0)


# In[38]:


clf=DecisionTreeClassifier()
tra=clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)


# In[39]:


y_pred


# In[40]:


print(confusion_matrix(y_test,y_pred))
print(clf.score(x_test,y_test))


# In[41]:


#from sklearn.externals.six import StringIO
#from IPython.display import Image
#from sklearn.tree import export_graphviz
#import pydotplus
#dot_data=StringIO()
#export_graphviz(clf, out_file=dot_data,filled=True, rounded=True, special_characters=True)
#graph=pydotplus.graph_from_dot_data(dot_data.getvalue())
#Image(graph.create_png())


# In[ ]:




