#!/usr/bin/env python
# coding: utf-8

# In[126]:


#Importing all the necessary libraries
import numpy as np
import pandas as pd
import time
from cvxopt import solvers
from cvxopt import matrix
from scipy.spatial.distance import pdist, cdist, squareform
from sklearn.svm import SVC
from multiprocessing import Process


# In[27]:


#Importing data
s = time.time()
data = pd.read_csv('../fashion_mnist/train.csv', header=None)
data = data.to_numpy()
print("Data imported in: ", time.time()-s, "s")
C = 1
gamma = 0.05


# In[28]:


#Function to extract subset of data, for two desired classes
def extract_data(d, d_prime):
    
    index_d = np.argwhere(data[:,-1]==d)[:,0]
    index_dp = np.argwhere(data[:,-1]==d_prime)[:,0]
    
    count1 = index_d.shape[0]
    count2 = index_dp.shape[0]
    
    return count1, count2, np.concatenate( (data[index_d, :-1], data[index_dp, :-1]), axis = 0)


# In[55]:


#Function to train a binary classifier for two desired classes
#Returns the learnt value of alphas and intercept b
def binary_classifier(d, dp):
    
    count_d, count_dp, data_d_dp = extract_data(d, dp)
    data_d_dp /= 255

    y_d = np.full(count_d, -1).reshape(-1,1)
    y_dp = np.full(count_dp, 1).reshape(-1,1)

    y = np.concatenate((y_d, y_dp), axis = 0)
    y_d = []
    y_dp = []
    
    s = time.time()
    P_g = squareform(pdist(data_d_dp))
    P_g = np.exp(-gamma*np.square(P_g))
    P_gaussian = matrix(np.multiply(np.matmul(y, y.T), P_g))
    print("Kernel Computed for ", d," & ", dp, " in ", time.time()-s,"s")
    
    q = matrix(-np.ones((count_d+count_dp, 1)))
    G = matrix(np.concatenate((np.eye(count_d+count_dp)*-1,np.eye(count_d+count_dp)), axis = 0))
    h = matrix(np.concatenate((np.zeros(count_d+count_dp), np.ones(count_d+count_dp) * C), axis = None))
    b_solver = matrix(np.zeros(1))
    A = matrix(y.reshape(1, -1), tc='d')
    
    s = time.time()
    solution_gaussian = solvers.qp(P_gaussian, q, G, h, A, b_solver)
    print("Gaussian Solved for ", d," & ", dp, " in ", time.time()-s,"s")
    alphas_gaussian = np.array(solution_gaussian['x'])
    
    alpha_y = (y * alphas_gaussian)
    b_0 = np.max(np.matmul(P_g[:count_d,:], alpha_y))
    b_1 = np.min(np.matmul(P_g[count_d:,:], alpha_y))
    P_g = []
    b = (-b_0-b_1)/2
    
    return (alphas_gaussian, b)


# In[66]:


#Enumerating the desired models of binary classifiers
n=10
ids = []
for i in range(n):
    for j in range(i+1, n):
        ids.append((i,j))

alphas = []
b = np.zeros((n,n))


# In[68]:


batchsize = 1
#Training 45 different models of binary classifiers
#Tried doing it with help of multiprocessing but was of no use, since the CVXOPT solver itself uses 4 threads.
s1 = time.time()
for i in range(int(45/batchsize)):
    
    print("Going for batch ", i)
    
    s = time.time()
    for j in range(batchsize):
        
        d, dp = ids[i*batchsize + j]
        
        alphas_g, b_g = binary_classifier(d,dp)
        
        alphas.append((alphas_g, d, dp))
        b[d, dp] = b_g
#         p = Process(target=binary_classifier, args=(n, m))
#         p.start()
#         processes.append(p)
#     for p in processes:
#         p.join()
    print("Done for batch ", i, " in ", time.time()-s)
print("One vs One Done in ", time.time()-s1, "s")


# In[109]:


#Saving the learnt values of alphas for all the 45 classifiers for future use
f = "alpha_b_ovo.csv"
f = open(f,"w")
count = 0
for i in range(len(alphas)):
    alpha, d, dp = alphas[i]
    alpha = ",".join(alpha.flatten().astype("str"))
    alpha = str(d),str(dp),str(b[d][dp]),alpha
    alpha = ",".join(alpha)
    f.write(alpha+"\n")


# In[111]:


#Importing Validation and testing data
s = time.time()

data_val = pd.read_csv('../fashion_mnist/val.csv', header=None)
data_val = data_val.to_numpy()

data_test = pd.read_csv('../fashion_mnist/test.csv', header=None)
data_test = data_test.to_numpy()

y_val = data_val[:,-1]
y_test = data_test[:,-1]

data_val = data_val[:,:-1]
data_test = data_test[:,:-1]

data_val /= 255
data_test /= 255
print("Validation and Test Data imported in: ", time.time()-s, "s")


# In[183]:


#Function to predict classes of test set, using One vs One Classifier technique
def prediction(alphas, b, test):
    n_test = test.shape[0]
    scores = np.zeros((n_test, 10))
    votes = np.zeros((n_test, 10))
    
    for i in alphas:
        s = time.time()
        alpha, d, dp = i
        
        count_d, count_dp, data_d_dp = extract_data(d, dp)
        data_d_dp /= 255

        y_d = np.full(count_d, -1).reshape(-1,1)
        y_dp = np.full(count_dp, 1).reshape(-1,1)

        y = np.concatenate((y_d, y_dp), axis = 0)
        y_d = []
        y_dp = []
        alpha_y = (y * alpha)
        
        pred = cdist(data_d_dp, test)
        pred = np.exp(-gamma*np.square(pred))
        pred = np.sum(np.multiply(pred, alpha_y), axis = 0)  + b[d][dp]
        
        index_d = np.argwhere(pred<0).flatten()
        index_dp = np.argwhere(pred>=0).flatten()
        
        votes[index_d,d] += 1
        votes[index_dp,dp] += 1
        
        scores[index_d,d] += abs(pred[index_d])
        scores[index_dp,dp] += abs(pred[index_dp])
        print("Done for ",d,"&",dp,"in :", time.time()-s)
        
        
        
        
    class_pred = []
    for i in range(n_test):
        vote = votes[i]
        winner = np.argwhere(vote == np.max(vote))
        if(winner.shape[0]>1):
            winner = winner.flatten()
            score = scores[i]
            won = np.argwhere(score == np.max(score[winner]))
            class_pred.append(won[0][0])
        else:
            class_pred.append(winner[0][0])
            
    return class_pred


# In[178]:


#Accuracy over a given preediction set and set of actual labels
def accuracy(pred, y):
    confusion = np.zeros((10, 10))
    count = 0
    for i in range(len(y)):
        confusion[int(pred[i]), int(y[i])] += 1
        if(pred[i]==y[i]):
            count += 1
    return count*100/(len(y)), confusion 


# In[180]:


#Accuracy over validation set in our implementation
s = time.time()
pred_val = prediction(alphas, b, data_val)
print("Time taken to predict Validation set is: ", time.time()-s)
acc_val, confusion_val = accuracy(pred_val, y_val)
print("Accuarcy over validation set in multi class classification is: ", acc_val, "%")


# In[185]:


#Accuracy over test set in our implementation
s = time.time()
pred_test = prediction(alphas, b, data_test)
print("Time taken to predict test set is: ", time.time()-s)
acc_test, confusion_test = accuracy(pred_test, y_test)
print("Accuarcy over test set in multi class classification is: ", acc_test, "%")


# In[190]:


#Confusion matrix for test and validation set in our implementation
print("The confusion matrix for validation set is:\n", confusion_val,"\n")
print("The confusion matrix for test set is:\n", confusion_test)


# In[198]:


#Enumerating the desired models of binary classifiers
n=10
ids = []
for i in range(n):
    for j in range(i+1, n):
        ids.append((i,j))
        
#One vs One SVM Sklearn
#Training 45 different models of binary classifiers using Sklearn library
batchsize = 1
clfs = []
s1 = time.time()
for i in range(int(45/batchsize)):
    
    print("Going for batch ", i)
    
    s = time.time()
    for j in range(batchsize):
        
        d, dp = ids[i*batchsize + j]
        count_d, count_dp, data_d_dp = extract_data(d, dp)
        data_d_dp /= 255

        y_d = np.full(count_d, -1).reshape(-1,1)
        y_dp = np.full(count_dp, 1).reshape(-1,1)

        y = np.concatenate((y_d, y_dp), axis = 0)
        y_d = []
        y_dp = []
        
        clf = SVC(kernel='rbf',gamma=0.05)
        clf.fit(data_d_dp, y.flatten())
        
        clfs.append((clf,d,dp))
    print("Done for batch ", i, " in ", time.time()-s)
print("One vs One SKlearn Done in ", time.time()-s1, "s")


# In[203]:


#Function to predict classes of test set, using One vs One Classifier technique in all the 45 classifiers
def prediction_sklearn(clfs, test):
    n_test = test.shape[0]
    scores = np.zeros((n_test, 10))
    votes = np.zeros((n_test, 10))
    
    for i in clfs:
        s = time.time()
        clf, d, dp = i
        
        pred = clf.decision_function(test)
        index_d = np.argwhere(pred<0).flatten()
        index_dp = np.argwhere(pred>=0).flatten()
        
        votes[index_d,d] += 1
        votes[index_dp,dp] += 1
        
        scores[index_d,d] += abs(pred[index_d])
        scores[index_dp,dp] += abs(pred[index_dp])
        print("Done for ",d,"&",dp,"in :", time.time()-s)
        
        
        
        
    class_pred = []
    for i in range(n_test):
        vote = votes[i]
        winner = np.argwhere(vote == np.max(vote))
        if(winner.shape[0]>1):
            winner = winner.flatten()
            score = scores[i]
            won = np.argwhere(score == np.max(score[winner]))
            class_pred.append(won[0][0])
        else:
            class_pred.append(winner[0][0])
            
    return class_pred


# In[204]:


#Accuracy over validation set in Sklearn implementation
s = time.time()
pred_val_sklearn = prediction_sklearn(clfs, data_val)
print("Time taken to predict Validation set is: ", time.time()-s)
acc_val_sklearn, confusion_val_sklearn = accuracy(pred_val_sklearn, y_val)
print("Accuarcy over validation set in multi class classification is: ", acc_val_sklearn, "%")


# In[205]:


#Accuracy over test set in Sklearn implementation
s = time.time()
pred_test_sklearn = prediction_sklearn(clfs, data_test)
print("Time taken to predict test set is: ", time.time()-s)
acc_test_sklearn, confusion_test_sklearn = accuracy(pred_test_sklearn, y_test)
print("Accuarcy over validation set in multi class classification is: ", acc_test_sklearn, "%")


# In[206]:


#Confusion matrix for test and validation set in Sklearn's implementation
print("The confusion matrix for test set in sklearn is:\n", confusion_val_sklearn,"\n")
print("The confusion matrix for test set in sklearn is:\n", confusion_test_sklearn)

