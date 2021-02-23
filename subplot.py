#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt


# In[2]:


f=open('salesdata2.csv','r')
salesdata=f.readlines()


# In[3]:


x_days=[1,2,3,4,5]
y_price1=[9,9.5,10.1,10,11]
y_price2=[12,13,14,15,15.5]
sale_list=[]
s_list=[]
c_list=[]


# In[4]:


for records in salesdata:
    sale,cost=records.split(',')
    s_list.append(int(sale))
    c_list.append(int(cost))


# In[5]:


sale_list.append(s_list)
sale_list.append(c_list)


# In[19]:


plt.subplot(2,3,1)
plt.title("Sale vs Cost")
plt.xlabel('Sale')
plt.ylabel('Cost')
plt.scatter(s_list,c_list,marker='*',s=90,c='b')

plt.subplot(2,3,2)
plt.title("Sales in USD")
plt.ylabel("USD")
plt.boxplot(sale_list,
            patch_artist=True,
            boxprops=dict(facecolor='g', color='r', linewidth=2),
            whiskerprops=dict(color='r', linewidth=2), 
            medianprops=dict(color='w', linewidth=1),
            capprops=dict(color='k', linewidth=2), 
            flierprops=dict(markerfacecolor='r',marker='o',markersize=7)
           )

plt.subplot(2,3,3)
plt.title("Histogram of Sales")
plt.ylabel('USD')
plt.hist(s_list,bins=8,rwidth=0.9, color='c')

plt.subplot(2,3,4)
plt.title("Stockprice")
plt.xlabel("Day")
plt.ylabel("Prices")
plt.plot(x_days,y_price1,label='STOCK1',color='green',marker='*',markersize=10,linewidth=3,linestyle='--')
plt.plot(x_days,y_price2,label='STOCK2',color='red',marker='o',markersize=10,linewidth=3,linestyle='-')
plt.legend(loc=2, fontsize=8)

plt.show()


# In[ ]:





# In[ ]:




