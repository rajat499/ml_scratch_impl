#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing all the necessary libraries
import numpy as np
import pandas as pd
import time
from cvxopt import solvers
from cvxopt import matrix
from scipy.spatial.distance import pdist, cdist, squareform
from sklearn.svm import SVC


# In[2]:


#Function to extract subset of data, for two desired classes
def extract_data(data, d, d_prime):
    
    index_d = np.argwhere(data[:,-1]==d)[:,0]
    index_dp = np.argwhere(data[:,-1]==d_prime)[:,0]
    
    count1 = index_d.shape[0]
    count2 = index_dp.shape[0]
    
    return count1, count2, np.concatenate( (data[index_d, :-1], data[index_dp, :-1]), axis = 0)


# In[3]:


#Importing data
s = time.time()
data = pd.read_csv('../fashion_mnist/train.csv', header=None)
data = data.to_numpy()
print("Data imported in: ", time.time()-s, "s")

count_5, count_6, data_5_6 = extract_data(data, 5, 6)
data_5_6 /= 255

y_5 = np.full(count_5, -1).reshape(-1,1)
y_6 = np.full(count_6, 1).reshape(-1,1)

y = np.concatenate((y_5, y_6), axis = 0)
y_5 = []
y_6 = []


# In[4]:


#Value of C
C = 1
#Formulating the parameters of CVXOPT Quadratic Problem solver for Linear Kernel
X = np.multiply(y, data_5_6)
P = matrix(np.matmul(X, X.T))
q = matrix(-np.ones((count_5+count_6, 1)))
G = matrix(np.concatenate((np.eye(count_5+count_6)*-1,np.eye(count_5+count_6)), axis = 0))
h = matrix(np.concatenate((np.zeros(count_5+count_6), np.ones(count_5+count_6) * C), axis = None))
b_solver = matrix(np.zeros(1))
A = matrix(y.reshape(1, -1), tc='d')


# In[5]:


#Solving for alphas in Linear Kernel
s = time.time()
solution = solvers.qp(P, q, G, h, A, b_solver)
print("Solved in ", time.time()-s,"s")
alphas = np.array(solution['x'])


# In[33]:


#Support Vectors in Linear Kernel
np.argwhere(alphas>0.0000001).shape


# In[26]:


#Getting value of w and b from alpha
w = (np.matmul((y * alphas).T , data_5_6)).reshape(-1,1)
b_0 = np.max(np.matmul(data_5_6[:count_5,:], w))
b_1 = np.min(np.matmul(data_5_6[count_5:,:], w))
b_intercept = (-b_0-b_1)/2


# In[27]:


#Function to calculate accuracy of a given prediction
def accuracy(pred, y):
    count = 0
    for i in range(pred.shape[0]):
        if(pred[i][0]==y[i][0]):
            count += 1
    
    return count*100/(pred.shape[0])

#Function to predict classes for a given test set x in Linear Kernel, w and b are passed as parameters
def prediction(w, b, x):
    theta = np.matmul(x,w) + b
    theta[theta<0] = -1
    theta[theta>=0] = 1
    return theta


# In[28]:


#Importing Validation and testing data
s = time.time()

data_val = pd.read_csv('../fashion_mnist/val.csv', header=None)
data_val = data_val.to_numpy()

data_test = pd.read_csv('../fashion_mnist/test.csv', header=None)
data_test = data_test.to_numpy()

count_val5, count_val6, data_val = extract_data(data_val, 5, 6)
count_test5, count_test6, data_test = extract_data(data_test, 5, 6)

data_val /= 255
data_test /= 255
print("Validation and Test Data imported in: ", time.time()-s, "s")


# In[29]:


#Accuracy of Validation set in Linear Kernel

y_val5 = np.full(count_val5, -1).reshape(-1,1)
y_val6 = np.full(count_val6, 1).reshape(-1,1)

y_val = np.concatenate((y_val5, y_val6), axis = 0)
y_val5 = []
y_val6 = []

pred_val = prediction(w, b_intercept, data_val)
print("Accuarcy over validation set is: ", accuracy(pred_val, y_val), "%")


# In[30]:


#Accuracy of Test set in Linear kernel

y_test5 = np.full(count_test5, -1).reshape(-1,1)
y_test6 = np.full(count_test6, 1).reshape(-1,1)

y_test = np.concatenate((y_test5, y_test6), axis = 0)

pred_test = prediction(w, b_intercept, data_test)
print("Accuarcy over test set is: ", accuracy(pred_test, y_test), "%")


# In[49]:


#The value of W and intercept b in our implementation of Linear Kernel
w, b_intercept


# In[31]:


#Formulating the parameters of CVXOPT Quadratic Problem solver for Gaussian Kernel
s = time.time()
P_g = squareform(pdist(data_5_6))
P_g = np.exp(-0.05*np.square(P_g))
P_gaussian = matrix(np.multiply(np.matmul(y, y.T), P_g))
print("Kernel Computed in ", time.time()-s,"s")


# In[34]:


#Solving for alphas in Gaussian Kernel
s = time.time()
solution_gaussian = solvers.qp(P_gaussian, q, G, h, A, b_solver)
print("Gaussian Solved in ", time.time()-s,"s")
alphas_gaussian = np.array(solution_gaussian['x'])


# In[43]:


#Support Vectors of Gaussian Kernel
np.argwhere(alphas_gaussian>0.0000001).shape


# In[44]:


#Solving for alphas in Gaussian Kernel
alpha_y = (y * alphas_gaussian)
b_0 = np.max(np.matmul(P_g[:count_5,:], alpha_y))
b_1 = np.min(np.matmul(P_g[count_5:,:], alpha_y))
b_intercept_g = (-b_0-b_1)/2


# In[45]:


#Function to predict classes in Gaussian Kernel, alpha, b and training data set is passed as parameters 
def prediction_g(alpha_y, data, test, b, gamma):
    pred = cdist(data, test)
    pred = np.exp(-gamma*np.square(pred))
    pred = np.sum(np.multiply(pred, alpha_y), axis = 0)  + b
    pred[pred>=0] = 1
    pred[pred<0] = -1
    return pred

def accuracy_g(pred, y):
    count = 0
    for i in range(len(pred)):
        if(pred[i]==y[i][0]):
            count += 1
    return count*100/len(pred)


# In[67]:


#Value of alpha_y and intercept b in our implementation of Gaussian Kernel
alpha_y, b_intercept_g


# In[52]:


#Accuracy of Validation set in Gaussian Kernel
pred_val_g = prediction_g(alpha_y, data_5_6, data_val, b_intercept_g, 0.05)
print("Accuarcy over validation set in gaussian is: ", accuracy_g(pred_val_g, y_val), "%")


# In[53]:


#Accuracy of Test set in Gaussian Kernel
pred_test_g = prediction_g(alpha_y, data_5_6, data_test, b_intercept_g, 0.05)
print("Accuarcy over test set in gaussian is: ", accuracy_g(pred_test_g, y_test), "%")


# In[54]:


#Training two different models of Sklearn's SVM using both linear and gaussian kernel
clf_l = SVC(kernel='linear')
clf_g = SVC(kernel='rbf',gamma=0.05)

s = time.time()
clf_l.fit(data_5_6, y.flatten())
print("Linear model trained in ", time.time()-s, "s")

s = time.time()
clf_g.fit(data_5_6, y.flatten())
print("Gaussian model trained in ", time.time()-s, "s")

val_sklearn_linear = clf_l.score(data_val, y_val.flatten())
test_sklearn_linear = clf_l.score(data_test, y_test.flatten())

val_sklearn_gaussian = clf_g.score(data_val, y_val.flatten())
test_sklearn_gaussian = clf_g.score(data_test, y_test.flatten())


# In[59]:


#The value of W and intercept b in Sklearn's SVM implementation using Linear Kernel
clf_l.coef_[0], clf_l.intercept_


# In[66]:


#Value of alpha_y and intercept b in Sklearn's SVM implementation using Gaussian Kernel
clf_g.dual_coef_, clf_g.intercept_


# In[70]:


#Support Vectors in Sklearn's SVM implementations
(clf_l.support_vectors_).shape, (clf_g.support_vectors_).shape 


# In[62]:


#Accuracy over Validation and test sets in Sklearn's implmentations of linear and gaussian kernels

print("Accuarcy over validation set in sklearn_linear is: ", 100*val_sklearn_linear, "%")
print("Accuarcy over test set in sklearn_linear is: ", 100*test_sklearn_linear, "%")
print("Accuarcy over validation set in sklearn_gaussian is: ", 100*val_sklearn_gaussian, "%")
print("Accuarcy over test set in sklearn_gaussian is: ", 100*test_sklearn_gaussian, "%")

