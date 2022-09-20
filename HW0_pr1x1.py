#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[7]:


df = pd.read_csv ('https://raw.githubusercontent.com/Owen3vans/4105HW0/main/D3.csv')
df.head()
M=len(df) ##length of data set
M         ## verifies correct data set


# In[38]:


x= df.values[:,0] #x1 values
y= df.values[:,3] #y values
m= len(x) #number of training values could be x or y
m #verify number of entries 


# In[39]:


print('X = ', x[: 5]) #first 5 values of x and y for verification
print('Y = ', y[: 5])


# In[40]:


x0 = np.ones((m,1)) #create matrix all 1s
x0[:5]


# In[41]:


x1 = x.reshape(m, 1) #create matrix with x1 values 
x1[:5]


# In[54]:


x = np.hstack((x0,x1)) #combines x0 x1 array
x[:5]


# In[55]:


theta = np.zeros(2) #theta value set to zero
theta


# In[56]:


def compute_cost(x,y,theta):
    predictions = x.dot(theta)
    errors = np.subtract(predictions, y)
    sqrErrors = np.square(errors)
    J= 1/(2*m)*np.sum(sqrErrors)
    return J


# In[57]:


cost = compute_cost(x,y,theta)
print(cost)


# In[67]:


def Gradiant_decent (x,y,theta,alpha,itterations):
    cost_history= np.zeros(itterations)
    for i in range(itterations):
        predictions= x.dot(theta)
        errors= np.subtract(predictions, y)
        sum_delta= (alpha / m)* x.transpose().dot(errors);
        theta = theta - sum_delta;
        cost_history[i]= compute_cost(x,y,theta)
    return theta, cost_history


# In[107]:


theta =[0.,0.] #set theta to 0
itterations = 1500;
alpha=0.1;


# In[111]:


theta, cost_history = Gradiant_decent(x,y,theta,alpha,itterations)
print('Final value of theta= ' ,theta)
print('cost history= ', cost_history)


# In[109]:


plt.scatter(x[:,1], y, color='red', marker= '+', label= 'Training Data') 
plt.plot(x[:,1],x.dot(theta), color='green', label='Linear Regression')
plt.rcParams["figure.figsize"] = (10,6) 
plt.grid() 
plt.xlabel('X1') 
plt.ylabel('y') 
plt.title('Linear Regression Fit') 
plt.legend() 


# In[110]:


plt.plot(range(1, itterations + 1),cost_history, color='blue') 
plt.rcParams["figure.figsize"] = (10,6) 
plt.grid() 
plt.xlabel('Number of iterations') 
plt.ylabel('Cost (J)') 
plt.title('Convergence of gradient descent') 


# In[ ]:


#alpha value of 0.1 gives lowest loss the quickest with cost =1 at 100 itterations
#alpha value of 0.01 gives cost= 1 at 1100 iterations
#alpha = 0.05 gives cost= 1 at 250 itterations

