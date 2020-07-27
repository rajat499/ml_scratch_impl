import re
import csv
import os
import numpy as np
from math import log

filename = "training_noemoticon.csv"

#count of words by classes(frequency)
dictpos={}
dictneg={}

#total words in class 0 and 4 with frequency
totalwords={}

#count of no. of classes
countpos=0
countneg=0

with open(filename, 'r',encoding='latin-1') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        l=re.findall(r"[\w']+", row[5])
        if(int(row[0])!=2):
            for e in l:
                if e in totalwords:
                    totalwords[e]+=1
                else:
                    totalwords[e]=1
        if(int(row[0])==0):
            countneg+=1
            for e in l:
                if e in dictneg:
                    dictneg[e]+=1
                else:
                    dictneg[e]=1
        if(int(row[0])==4):
            countpos+=1
            for e in l:
                if e in dictpos:
                    dictpos[e]+=1
                else:
                    dictpos[e]=1


file2="test.csv"
with open(file2, 'r',encoding='latin-1') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        if(int(row[0])!=2):
            l=re.findall(r"[\w']+", row[5])
            for e in l:
                if e not in totalwords:
                    totalwords[e]=1

#total no. of words in each class of test data
totalpos=sum(dictpos.values()) 
totalneg=sum(dictneg.values())
# total=totalneg+totalpos

#Size of dictionary; total no. of distinct words
total=len(totalwords)
print(total)
ppos=log(1/(totalpos+total))
pneg=log(1/(totalneg+total))

probpos={}
probneg={}

#count for classes
count=countneg+countpos
p0=log(countneg/count)
p1=log(countpos/count)

for e in totalwords:
    if e in dictpos:
        probpos[e]=log((dictpos[e]+1)/(totalpos+total))
    else:
        probpos[e]=(ppos)
    if e in dictneg:
        probneg[e]=log((dictneg[e]+1)/(totalneg+total))
    else:
        probneg[e]=(pneg)

crightnb=0
ctot=0
czero=0
crightrand=0
pred=0

#0-class 0, 1-class 4
confusionmatrix=np.zeros([2,2],dtype=int)
with open(file2, 'r',encoding='latin-1') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        if(int(row[0])!=2):
            ctot+=1
            if(int(row[0])==0):
                czero+=1
            k=int.from_bytes(os.urandom(8), byteorder="big") / ((1 << 64) - 1)
            t=0
            if(k<0.5):
                t=0
            else:
                t=4
            l=re.findall(r"[\w']+", row[5])
            sum0=p0
            sum1=p1
            for e in l:
                sum0+=probneg[e]
                sum1+=probpos[e]
            
            if(sum0>sum1):
                pred=0
            else:
                pred=4
            if(int(row[0])==pred):
                crightnb+=1
                if(int(row[0])==0):
                    confusionmatrix[0,0]+=1
                else:
                    confusionmatrix[1,1]+=1
            else:
                if(int(row[0])==0):
                    confusionmatrix[1,0]+=1
                else:
                    confusionmatrix[0,1]+=1
            if(int(row[0])==t):
                crightrand+=1

print("Random Prediction Accuracy :",crightrand/ctot)
print("Majority Prediction Accuracy :",max(czero,ctot-czero)/ctot)
print("Naive Bayes Accuracy", crightnb/ctot)
print(confusionmatrix)
print(ctot)
            
