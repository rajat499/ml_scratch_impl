#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Importing all the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import time
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectPercentile
from sklearn import metrics


# In[3]:


#Importing data
data = pd.read_csv('../data/training_noemoticon.csv', header=None, encoding='latin-1')
data = data.to_numpy()


# In[6]:


#Stemming and Removing Stop Words using nltk

#Function to preprocess the input x, for removing special characters and splitting the entire string
#Does stemming and removes stop words
#Also converts the entire text to lower case.

stop_words = set(stopwords.words('english'))
ps = PorterStemmer()
def process(x):
    l = []
    for i in x:
        t = (re.split("[^a-z0-9'@]", i.lower()))
        prime = []
        for j in t:
            j=ps.stem(j)
            if(j=='' or j in stop_words):
                continue
            if(j[0]=='@'):
                continue
            prime.append(j)
        l.append(prime)
    return l


# In[3]:


#Segregating input data on the basis of their actual classes.

index_0 = np.argwhere(data[:,0]==0)[:,0]
index_1 = np.argwhere(data[:,0]==4)[:,0]

x_0 = data[index_0, 5]
y_0 = data[index_0, 0].reshape(-1,1)
count_0 = index_0.shape[0]

x_1 = data[index_1, 5]
y_1 = data[index_1, 0].reshape(-1,1)
count_1 = index_1.shape[0]

p_0 = np.log(count_0) - np.log(count_0 + count_1)
p_1 = np.log(count_1) - np.log(count_0 + count_1)

print(p_0, p_1)

print("Starting for class 0")
start = time.time()
x_0 = process(x_0)
elapsed = time.time()-start
print("Done for class 0, time elapsed is: ",elapsed, "\n")

print("Starting for class 1")
start = time.time()
x_1 = process(x_1)
elapsed = time.time()-start
print("Done for class 1, time elapsed is: ",elapsed, "\n")


# In[14]:


# #Saving preprocessed text for future use
# f = open("x_0.txt", "w")
# for i in range(len(y_0)):
#     f.write(str(y_0[i][0]))
#     for j in x_0[i]:
#         f.write(" "+j)
#     f.write("\n")
    
# f = open("x_1.txt", "w")
# for i in range(len(y_1)):
#     f.write(str(y_1[i][0]))
#     for j in x_1[i]:
#         f.write(" "+j)
#     f.write("\n")


# In[4]:


#Reading line fron x_0 and x_1, the preprocessed text stored
f = open("x_0.txt", "r")
y_0 = []
x_0 = []
for line in f:
    t = re.split(" ", line)
    y = int(t[0])
    t = t[1:]
    if(t!=[]):
        t[-1] = t[-1][:-1]
    y_0.append(y)
    x_0.append(t)
    
f = open("x_1.txt", "r")
y_1 = []
x_1 = []
for line in f:
    t = re.split(" ", line)
    y = int(t[0])
    t = t[1:]
    if(t!=[]):
        t[-1] = t[-1][:-1]
    y_1.append(y)
    x_1.append(t)

count_0 = len(y_0)
count_1 = len(y_1)

p_0 = np.log(count_0) - np.log(count_0 + count_1)
p_1 = np.log(count_1) - np.log(count_0 + count_1)

y_0 = np.array(y_0).reshape(-1,1)
y_1 = np.array(y_1).reshape(-1,1)


# In[7]:


#Importing test data
test = pd.read_csv('../data/test.csv', header=None, encoding='latin-1')
test = test.to_numpy()

#Segregating test data on the bases of their class, and taking only class 0 and 4
index_t = np.union1d(np.argwhere(test[:,0]==0)[:,0], np.argwhere(test[:,0]==4)[:,0])
x_test = test[index_t, 5]
y_test = test[index_t, 0].reshape(-1,1)
x_test = process(x_test)


# In[8]:


#Creating three diferent dictonaries, two for each classes and one for union of two
def add_dict(x_0, x_1):
    dict_t = {}
    dict_0 = {}
    dict_1 = {}
    for i in x_0:
        for j in i:
            if(j in dict_0):
                dict_0[j] = dict_0.get(j) + 1
            else:
                dict_0[j] = 1
            
            if(j in dict_t):
                dict_t[j] = dict_t.get(j) + 1
            else:
                dict_t[j] = 1
                
    
    
    for i in x_1:
        for j in i:
            if(j in dict_1):
                dict_1[j] = dict_1.get(j) + 1
            else:
                dict_1[j] = 1
            
            if(j in dict_t):
                dict_t[j] = dict_t.get(j) + 1
            else:
                dict_t[j] = 1
                
    return dict_t, dict_0, dict_1

dict_t, dict_0, dict_1 = add_dict(x_0, x_1)


# In[9]:


#Total no of features, and their count in respective classes
no_of_words = len(dict_t)
total_words_0 = sum(dict_0.values())
total_words_1 = sum(dict_1.values())
c = 1


# In[10]:


#Function to predict classes, and calculate accuracy of prediction.
#returns confusion matrix and scores of prediction as well
def accuracy(y, x):
    confusion = np.zeros([2,2])
    pred = []
    scores = []
    for i in range(len(x)):
        p0 = p_0
        p1 = p_1
        for j in x[i]:
            if(j in dict_0):
                p0 += np.log((dict_0[j]+1)) - np.log((total_words_0 + no_of_words*c))
            else:
                p0 += np.log(1) - np.log((total_words_0 + no_of_words*c))
                
            
            if(j in dict_1):
                p1 += np.log((dict_1[j]+1)) - np.log((total_words_1 + no_of_words*c))
            else:
                p1 += np.log(1) - np.log((total_words_1 + no_of_words*c))
        
        scores.append(np.exp(p1)/(np.exp(p0)+np.exp(p1)))
        if(p1>p0):
            pred.append(4)
        else:
            pred.append(0)
    
    total = len(pred)
    count_real = 0
    
    print(total)
    
    for i in range(len(pred)):
        if(pred[i]==int(y[i][0])):
            count_real += 1
            if(int(y[i][0])==0):
                confusion[0,0] += 1
            else:
                confusion[1,1] += 1
        else:
            if(int(y[i][0])==0):
                confusion[1,0] += 1
            else:
                confusion[0,1] += 1
                
    print(count_real)
    
    return count_real, total, confusion, scores


# In[11]:


#Accuracy over the test set
count_real_test, total_test, confusion_test, scores_test = accuracy(y_test, x_test)
print("Accuracy of the model over test set is: ", count_real_test*100/total_test)


# In[12]:


#Drawing ROC Curve on the basis of scores of prediction, after stopword removal and stemming

fpr, tpr, thresholds = metrics.roc_curve(np.asarray(y_test.flatten(), dtype=np.int64), scores_test, pos_label=4)
roc_auc = metrics.auc(fpr, tpr)
plt.figure()
lw=2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic(After stemming and stopword removal)')
plt.legend(loc="lower right")
plt.show()


# In[15]:


#Creating three diferent dictonaries for bigrams, two for each classes and one for union of two
def add_dict_bigram(x_0, x_1):
    
    dict_t = {}
    dict_0 = {}
    dict_1 = {}
    
    for i in x_0:
        
        for j in i:
            if(j in dict_0):
                dict_0[j] = dict_0.get(j) + 1
            else:
                dict_0[j] = 1
            
            if(j in dict_t):
                dict_t[j] = dict_t.get(j) + 1
            else:
                dict_t[j] = 1
        
        for item in nltk.bigrams(i):
            j = ' '.join(item)
            if(j in dict_0):
                dict_0[j] = dict_0.get(j) + 1
            else:
                dict_0[j] = 1
            
            if(j in dict_t):
                dict_t[j] = dict_t.get(j) + 1
            else:
                dict_t[j] = 1
                
    
    
    for i in x_1:
        
        for j in i:
            if(j in dict_1):
                dict_1[j] = dict_1.get(j) + 1
            else:
                dict_1[j] = 1
            
            if(j in dict_t):
                dict_t[j] = dict_t.get(j) + 1
            else:
                dict_t[j] = 1
        
        for item in nltk.bigrams(i):
            j = ' '.join(item)
            if(j in dict_1):
                dict_1[j] = dict_1.get(j) + 1
            else:
                dict_1[j] = 1
            
            if(j in dict_t):
                dict_t[j] = dict_t.get(j) + 1
            else:
                dict_t[j] = 1
                
    return dict_t, dict_0, dict_1

dict_t, dict_0, dict_1 = add_dict_bigram(x_0, x_1)


# In[16]:


#Total no of features, and their count in respective classes
no_of_words = len(dict_t)
total_words_0 = sum(dict_0.values())
total_words_1 = sum(dict_1.values())
c = 1


# In[17]:


#Function to predict classes, and calculate accuracy of prediction.
#returns confusion matrix and scores of prediction as well
def accuracy_bigram(y, x):
    confusion = np.zeros([2,2])
    pred = []
    scores = []
    for i in range(len(x)):
        
        p0 = p_0
        p1 = p_1
        
        for j in x[i]:
            if(j in dict_0):
                p0 += np.log((dict_0[j]+1)) - np.log((total_words_0 + no_of_words*c))
            else:
                p0 += np.log(1) - np.log((total_words_0 + no_of_words*c))
                
            
            if(j in dict_1):
                p1 += np.log((dict_1[j]+1)) - np.log((total_words_1 + no_of_words*c))
            else:
                p1 += np.log(1) - np.log((total_words_1 + no_of_words*c))
        
        for item in nltk.bigrams(x[i]):
            j = ' '.join(item)
            if(j in dict_0):
                p0 += np.log((dict_0[j]+1)) - np.log((total_words_0 + no_of_words*c))
            else:
                p0 += np.log(1) - np.log((total_words_0 + no_of_words*c))
                
            
            if(j in dict_1):
                p1 += np.log((dict_1[j]+1)) - np.log((total_words_1 + no_of_words*c))
            else:
                p1 += np.log(1) - np.log((total_words_1 + no_of_words*c))
        
        scores.append(np.exp(p1)/(np.exp(p0)+np.exp(p1)))
        if(p1>p0):
            pred.append(4)
        else:
            pred.append(0)
    
    total = len(pred)
    count_real = 0
    
    print(total)
    
    for i in range(len(pred)):
        if(pred[i]==int(y[i][0])):
            count_real += 1
            if(int(y[i][0])==0):
                confusion[0,0] += 1
            else:
                confusion[1,1] += 1
        else:
            if(int(y[i][0])==0):
                confusion[1,0] += 1
            else:
                confusion[0,1] += 1
                
    print(count_real)
    
    return count_real, total, confusion, scores


# In[28]:


#Accuracy over Class 0 of training set
accuracy_bigram(y_0, x_0)


# In[29]:


#Accuracy over class 4 of training set
accuracy_bigram(y_1, x_1)


# In[18]:


#Accuracy and Confusion matrix of test set after taking bigrams
count_real_test, total_test, confusion_test, scores_test = accuracy_bigram(y_test, x_test)
print("Accuracy of the model over test set in bigram is: ", count_real_test*100/total_test)
print(confusion_test)


# In[19]:


#Drawing ROC Curve on the basis of scores of prediction, for bigrams

fpr, tpr, thresholds = metrics.roc_curve(np.asarray(y_test.flatten(), dtype=np.int64), scores_test, pos_label=4)
roc_auc = metrics.auc(fpr, tpr)
plt.figure()
lw=2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for bigrams')
plt.legend(loc="lower right")
plt.show()


# In[20]:


#Creating three diferent dictonaries for trigrams, two for each classes and one for union of two
def add_dict_trigram(x_0, x_1):
    
    dict_t = {}
    dict_0 = {}
    dict_1 = {}
    
    for i in x_0:
        
        for j in i:
            if(j in dict_0):
                dict_0[j] = dict_0.get(j) + 1
            else:
                dict_0[j] = 1
            
            if(j in dict_t):
                dict_t[j] = dict_t.get(j) + 1
            else:
                dict_t[j] = 1
        
        for item in nltk.trigrams(i):
            j = ' '.join(item)
            if(j in dict_0):
                dict_0[j] = dict_0.get(j) + 1
            else:
                dict_0[j] = 1
            
            if(j in dict_t):
                dict_t[j] = dict_t.get(j) + 1
            else:
                dict_t[j] = 1
                
    
    
    for i in x_1:
        
        for j in i:
            if(j in dict_1):
                dict_1[j] = dict_1.get(j) + 1
            else:
                dict_1[j] = 1
            
            if(j in dict_t):
                dict_t[j] = dict_t.get(j) + 1
            else:
                dict_t[j] = 1
        
        for item in nltk.trigrams(i):
            j = ' '.join(item)
            if(j in dict_1):
                dict_1[j] = dict_1.get(j) + 1
            else:
                dict_1[j] = 1
            
            if(j in dict_t):
                dict_t[j] = dict_t.get(j) + 1
            else:
                dict_t[j] = 1
                
    return dict_t, dict_0, dict_1

dict_t, dict_0, dict_1 = add_dict_trigram(x_0, x_1)

#Total no of features, and their count in respective classes
no_of_words = len(dict_t)
total_words_0 = sum(dict_0.values())
total_words_1 = sum(dict_1.values())
c = 1


# In[23]:


#Function to predict classes, and calculate accuracy of prediction.
#returns confusion matrix and scores of prediction as well
def accuracy_trigram(y, x):
    confusion = np.zeros([2,2])
    pred = []
    scores = []
    for i in range(len(x)):
        
        p0 = p_0
        p1 = p_1
        
        for j in x[i]:
            if(j in dict_0):
                p0 += np.log((dict_0[j]+1)) - np.log((total_words_0 + no_of_words*c))
            else:
                p0 += np.log(1) - np.log((total_words_0 + no_of_words*c))
                
            
            if(j in dict_1):
                p1 += np.log((dict_1[j]+1)) - np.log((total_words_1 + no_of_words*c))
            else:
                p1 += np.log(1) - np.log((total_words_1 + no_of_words*c))
        
        for item in nltk.trigrams(x[i]):
            j = ' '.join(item)
            if(j in dict_0):
                p0 += np.log((dict_0[j]+1)) - np.log((total_words_0 + no_of_words*c))
            else:
                p0 += np.log(1) - np.log((total_words_0 + no_of_words*c))
                
            
            if(j in dict_1):
                p1 += np.log((dict_1[j]+1)) - np.log((total_words_1 + no_of_words*c))
            else:
                p1 += np.log(1) - np.log((total_words_1 + no_of_words*c))
        
        scores.append(np.exp(p1)/(np.exp(p0)+np.exp(p1)))  
        if(p1>p0):
            pred.append(4)
        else:
            pred.append(0)
    
    total = len(pred)
    count_real = 0
    
    print(total)
    
    for i in range(len(pred)):
        if(pred[i]==int(y[i][0])):
            count_real += 1
            if(int(y[i][0])==0):
                confusion[0,0] += 1
            else:
                confusion[1,1] += 1
        else:
            if(int(y[i][0])==0):
                confusion[1,0] += 1
            else:
                confusion[0,1] += 1
                
    print(count_real)
    
    return count_real, total, confusion, scores


# In[24]:


#Accuracy and Confusion matrix of test set after taking trigrams
count_real_test, total_test, confusion_test, scores_test = accuracy_trigram(y_test, x_test)
print("Accuracy of the model over test set in trigram is: ", count_real_test*100/total_test)
print(confusion_test)


# In[25]:


#Drawing ROC Curve on the basis of scores of prediction, for trigrams

fpr, tpr, thresholds = metrics.roc_curve(np.asarray(y_test.flatten(), dtype=np.int64), scores_test, pos_label=4)
roc_auc = metrics.auc(fpr, tpr)
plt.figure()
lw=2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for trigrams')
plt.legend(loc="lower right")
plt.show()


# In[19]:


#Accuracy over Class 0 of training set
accuracy_trigram(y_0, x_0)


# In[20]:


#Accuracy over Class 4 of training set
accuracy_trigram(y_1, x_1)


# In[8]:


#Creating Strings for passing to Tfidf vectorizer of sklearn
#The words are already preprocesses
corpus = []

for i in x_0:
    corpus.append(' '.join(i))

for i in x_1:
    corpus.append(' '.join(i))
    
test_x = []
for i in x_test:
    test_x.append(" ".join(i))
    
x_0=[]
x_1=[]


# In[9]:


#Calculating Termfrequency-InverseDocumentFrequency from sklearn library
vectorizer = TfidfVectorizer(stop_words='english')
train = vectorizer.fit_transform(corpus).astype(np.float32)
test = vectorizer.transform(test_x).astype(np.float32)
y_train = np.concatenate([y_0.flatten(), y_1.flatten()])


# In[19]:


#Batch wise training of the all th features
n = len(corpus)
batch = 500
clf = GaussianNB()

start = time.time()
for i in range(int(n/batch)):
    s = i*batch
    e = (i+1)*batch
    print("Going for batch ", i)
    s1 = time.time()
    clf.partial_fit(train[s:e].toarray(), y_train[s:e], np.unique(y_train))
    print("Batch ", i, " done in ", time.time()-s1, "s")
    
print("Training time: ", time.time()-start)


# In[40]:


#Accuracy over test set after training for all the features
accuracy = clf.score(test.toarray(), np.asarray(y_test.flatten(), dtype=np.int64))
print("Accuracy of test set: ", accuracy)


# In[56]:


#Selecting the top-p percentile features 
p = 0.1 
select = SelectPercentile(f_classif, percentile=p)
select.fit(train, y_train)
train_select = select.transform(train)


# In[59]:


#Batch wise training of top-p percentile features
n = len(corpus)
batch = 1000
clf2 = GaussianNB()
test_select = select.transform(test)

start = time.time()
for i in range(int(n/batch)):
    s = i*batch
    e = (i+1)*batch
    s1 = time.time()
    clf2.partial_fit(train_select[s:e].toarray(), y_train[s:e], np.unique(y_train))
    
print("Training time: ", time.time()-start)


# In[60]:


#Accuracy over test set after training for top-p percentile features

start = time.time()
accuracy = clf2.score(test_select.toarray(), np.asarray(y_test.flatten(), dtype=np.int64))
print("Prediction time(test): ", time.time()-start)
print("Accuracy of test set: ", accuracy*100, ", for percentile selection = ", p)

