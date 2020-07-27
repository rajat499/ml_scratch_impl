#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing all the necessary libraries
import numpy as np
import pandas as pd
import time
from scipy.spatial.distance import pdist, cdist, squareform
from sklearn.svm import SVC
from multiprocessing import Process
import multiprocessing


# In[2]:


#Importing data
s = time.time()
data = pd.read_csv('../fashion_mnist/train.csv', header=None)
data = data.to_numpy()
y = data[:,-1]
data = data[:,:-1]
data /= 255
print("Data imported in: ", time.time()-s, "s")

#All the different desired values of C
C = [1e-5, 1e-3, 1, 5, 10]
gamma = 0.05

#Importing Validation set
data_val = pd.read_csv('../fashion_mnist/val.csv', header=None)
data_val = data_val.to_numpy()
y_val = data_val[:,-1]
data_val = data_val[:,:-1]
data_val /= 255

data = np.concatenate((data, data_val), axis = 0)
y = np.concatenate((y,y_val))


# In[3]:


#Dividing the training data into 5 different folds/sets
data_cross = []
y_cross = []

for i in range(5):
    data_i = data[i*5000:(i+1)*5000][:]
    y_i = y[i*5000:(i+1)*5000]
    data_cross.append(data_i)
    y_cross.append(y_i)
data = np.array(data_cross)
y = np.array(y_cross)

data_cross = []
y_cross = []


# In[4]:


#Function to train a Multi Class Sklearn's SVM modelf over data_x
# and then calculating accuracy over data_val and data_test
#This function will be used in Multiprocessing

def svm_ovo(data_x, data_y, data_val, y_val, data_test, y_test, c_acc, t_acc, index, C):
    print("Started SVM executing for C=",c, " index=", index)
    s = time.time()
    clf = SVC(C=C, kernel='rbf', gamma=0.05, decision_function_shape='ovo')
    clf.fit(data_x, data_y)
    c_acc[index] = clf.score(data_val, y_val)
    t_acc[index] = clf.score(data_test, y_test)
    print("C=",c," index=",index," val-",c_acc[index], "test - ",t_acc[index])
    print("Done SVM executing for C=",c, " index=", index," in ", time.time()-s)


# In[5]:


#Training a 5-fold model over data set for a particular value of C
#This function uses MULTIPROCESSING to train all the 5 models in parallel
#It tests all the 5 models over the test set data_test

def cross_fold(data, y, data_test, y_test, C):
    
    process = []
    manager = multiprocessing.Manager()
    c_acc = manager.dict()
    t_acc = manager.dict()
    for i in range(5):
        data_x = np.zeros((0,data.shape[2]))
        data_y = []
        for j in range(5):
            if(j==i):
                continue
            else:
                data_x = np.concatenate((data_x, data[j]), axis=0)
                data_y = np.concatenate((data_y, y[j]))
        
        p = Process(target=svm_ovo, args=(data_x, data_y, data[i], y[i], data_test, y_test, c_acc, t_acc, i, C))
        p.start()
        process.append(p)
        
    for p in process:
        p.join()
        
    cross_fold_acc = np.sum(c_acc.values())/len(c_acc)
    
    test_acc = np.sum(t_acc.values())/len(t_acc)
    
    return cross_fold_acc, test_acc
    


# In[6]:


#Importing Test Data
data_test = pd.read_csv('../fashion_mnist/test.csv', header=None)
data_test = data_test.to_numpy()
y_test = data_test[:,-1]
data_test = data_test[:,:-1]
data_test /= 255


# In[7]:


#Training models for different values of C and reporting their accuracies for comparison.
for c in C:
    print("Going for C=",c)
    s1 = time.time()
    cross_fold_acc, test_acc = cross_fold(data, y, data_test, y_test, c)
    print("Done for C=",c," in ",time.time()-s1)
    print("The reported 5-fold Cross validation accuarcy is ", cross_fold_acc*100,"% and test accuarcy is", test_acc*100,"%")


# In[ ]:




