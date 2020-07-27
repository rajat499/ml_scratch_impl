#!/usr/bin/env python
# coding: utf-8

# In[4]:


#Importing all the required libraries
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier as rf


# In[5]:


#A function to read the files which is given in sparse format
def read_files(x, y):
    x = '../ass3_parta_data/' + x
    f = open(x, 'r')
    line = f.readline().rstrip("\n").split(" ")
    num_samples, num_feat = int(line[0]), int(line[1])
    data_x = np.zeros((num_samples, num_feat))
    i = 0
    for line in f:
        line = line.split(" ")
        for value in line:
            value = value.split(":")
            data_x[i][int(value[0])] = float(value[1])
        i += 1
    
    data_y = np.genfromtxt('../ass3_parta_data/' + y, delimiter=' ')
    
    return data_x, data_y

s = time.time()
x_train, y_train = read_files('train_x.txt', 'train_y.txt')
x_val, y_val = read_files('valid_x.txt', 'valid_y.txt')
x_test, y_test = read_files('test_x.txt', 'test_y.txt')
print("Data imported in ", time.time()-s, "s")


# In[2]:


#Reading data using libraies

# s = time.time()
# x_train = data_utils.read_sparse_file('../ass3_parta_data/train_x.txt').todense()
# y_train = np.genfromtxt('../ass3_parta_data/train_y.txt', delimiter=' ')

# x_val = data_utils.read_sparse_file('../ass3_parta_data/valid_x.txt').todense()
# y_val = np.genfromtxt('../ass3_parta_data/valid_y.txt', delimiter=' ')

# x_test = data_utils.read_sparse_file('../ass3_parta_data/test_x.txt').todense()
# y_test = np.genfromtxt('../ass3_parta_data/test_y.txt', delimiter=' ')
# print("Data imported in ", time.time()-s, "s")


# In[3]:


#Getting all the parameter

n_estimators = np.arange(50, 451, 100)    
max_features = np.arange(0.1, 1.0, 0.2)
min_samples_split = np.arange(2, 11, 2)

#Parameter space over grid search has to be done
parameters = []
for i in n_estimators:
    for j in max_features:
        for k in min_samples_split:
            parameters.append((i,j,k))


# In[8]:


#Performing grid search for all the 125 models in the space

s1 = time.time()

#A dictionary that stores validation, test, train, and oob accuracy of the models learnt
models = {}
count = 0

for i in n_estimators:
    for j in max_features:
        
        s_ij = time.time()
        
        for k in min_samples_split:
            
            s = time.time()
            # A Sklearn Random forest model with given parameters and 6 jobs in parallel
            clf = rf(n_estimators = i, oob_score = True, max_features = j, min_samples_split = k, n_jobs = -2)
            clf.fit(x_train, y_train)
            val_acc = clf.score(x_val, y_val)
            test_acc = clf.score(x_test, y_test)
            train_acc = clf.score(x_train, y_train)
            e = time.time()
            
            print(clf.oob_score_, val_acc, test_acc)
            models[(i,j,k)] = (clf.oob_score_, val_acc, test_acc, train_acc)
            
            clf = None
            count += 1
            print("Time taken to train model", count,"with parameters ", i,j,k," is:", (e-s), "s")
        
        print("Done for parameters",i,j,"in", time.time()-s_ij,"s")

e1 = time.time()
print("Total time taken:", (e1-s1),"s")


# In[122]:


#Storing the info for future reference, so that time is saved in training again
f = "models.txt"
f = open(f,"w")

for r in models:
    m = " ".join(np.array(r).astype("str"))
    n = " ".join(np.array(models[r]).astype("str"))
    
    f.write(m+" "+n+"\n")


# In[88]:


#Selecting the optimal set of parameters, with the best OOB accuracy
optimal = None
best_oob = -1

for i in models:
    oob = models[i][1]
    if(oob >= best_oob):
        best_oob = oob
        optimal = i


# In[89]:


# Training the model with optimal set of accuarcy again
estimator, features, split = optimal

s = time.time()
optimal_model = rf(n_estimators = estimator, oob_score = True, max_features = features, min_samples_split = split, n_jobs = -2)
optimal_model.fit(x_train, y_train)
print("Time taken to train optimal model is: ", time.time()-s, "s")


# In[90]:


#Getting all the desired informations in part(a)
train_accuracy = optimal_model.score(x_train, y_train) * 100
oob_accuracy = models[optimal][0] * 100
val_accuracy = models[optimal][1] * 100
test_accuracy = models[optimal][2] * 100

print("The optimal model has", estimator, "no of estimators,", features*100,"% of max_features, and min_samples_split is", split)
print("The optimal model has Training accuracy =", train_accuracy,"%")
print("The optimal model has Out-of-bag accuracy =", oob_accuracy,"%")
print("The optimal model has Validation accuracy =", val_accuracy,"%")
print("The optimal model has Testing accuracy =", test_accuracy,"%")


# In[155]:


#Plotting the desired results with varying number of estimators
x_plot = []
val_plot = []
test_plot = []

#Getting data points from above dictionary
for i in n_estimators:
    x_plot.append(i)
    point = (i,features,split)
    val_plot.append(models[point][1])
    test_plot.append(models[point][2])

print("X_Plot", x_plot)
print("Val_Plot", val_plot)
print("Test_Plot", test_plot)

plt.figure()
lw=2
plt.plot(x_plot, val_plot, color='darkorange', lw=lw, label='Validation Accuracy')
plt.plot(x_plot, test_plot, color='navy', lw=lw, linestyle='--', label='Test Accuracy')
plt.xlim([0, 500])
plt.ylim([0.8, 0.81])
plt.ylabel('Accuracy')
plt.xlabel('Number of Estimators')
plt.title('Accuracy with varying number of estimators')
plt.legend(loc="lower right")
plt.show()


# In[156]:


#Plotting the desired results with varying max_features
x_plot = []
val_plot = []
test_plot = []

#Getting data points from above dictionary
for i in max_features:
    x_plot.append(i)
    point = (estimator, i, split)
    val_plot.append(models[point][1])
    test_plot.append(models[point][2])

print("X_Plot", x_plot)
print("Val_Plot", val_plot)
print("Test_Plot", test_plot)

plt.figure()
lw=2
plt.plot(x_plot, val_plot, color='darkorange', lw=lw, label='Validation Accuracy')
plt.plot(x_plot, test_plot, color='navy', lw=lw, linestyle='--', label='Test Accuracy')
plt.xlim([0, 1])
plt.ylim([0.8, 0.81])
plt.ylabel('Accuracy')
plt.xlabel('Number of Estimators')
plt.title('Accuracy with varying max_features')
plt.legend(loc="lower right")
plt.show()


# In[157]:


#Plotting the desired results with varying min_samples_split
x_plot = []
val_plot = []
test_plot = []

#Getting data points from above dictionary
for i in min_samples_split:
    x_plot.append(i)
    point = (estimator, features, i)
    val_plot.append(models[point][1])
    test_plot.append(models[point][2])

print("X_Plot", x_plot)
print("Val_Plot", val_plot)
print("Test_Plot", test_plot)

plt.figure()
lw=2
plt.plot(x_plot, val_plot, color='darkorange', lw=lw, label='Validation Accuracy')
plt.plot(x_plot, test_plot, color='navy', lw=lw, linestyle='--', label='Test Accuracy')
plt.xlim([1, 11])
plt.ylim([0.795, 0.81])
plt.ylabel('Accuracy')
plt.xlabel('Number of Estimators')
plt.title('Accuracy with varying min_samples_split')
plt.legend(loc="lower right")
plt.show()

