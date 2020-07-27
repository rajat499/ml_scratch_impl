#!/usr/bin/env python
# coding: utf-8

# In[74]:


#Importing all the required libraries
import numpy as np
import time
import matplotlib.pyplot as plt


# In[75]:


#A fucntion to read the files which is given in sparse format
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


# In[76]:


#Reading data using librraies

# s = time.time()
# x_train = data_utils.read_sparse_file('../ass3_parta_data/train_x.txt').todense()
# y_train = np.genfromtxt('../ass3_parta_data/train_y.txt', delimiter=' ')

# x_val = data_utils.read_sparse_file('../ass3_parta_data/valid_x.txt').todense()
# y_val = np.genfromtxt('../ass3_parta_data/valid_y.txt', delimiter=' ')

# x_test = data_utils.read_sparse_file('../ass3_parta_data/test_x.txt').todense()
# y_test = np.genfromtxt('../ass3_parta_data/test_y.txt', delimiter=' ')
# print("Data imported in ", time.time()-s, "s")


# In[77]:


#SKlearn implementation of decision tree
from sklearn import tree
s = time.time()
clf = tree.DecisionTreeClassifier(criterion='entropy')
m = clf.fit(x_train, y_train)
print("Accuracy over validation set sklearn: ", clf.score(x_val, y_val)*100, "%")
print("Accuracy over test set sklearn: ", clf.score(x_test, y_test)*100, "%")
print("Accuracy over training set sklearn: ", clf.score(x_train, y_train)*100, "%")
print("The depth of the decision tree is: ", clf.get_depth())
print("The number of leaf nodes in the tree is: ", clf.get_n_leaves())
print("Total number of nodes in the tree is: ", clf.tree_.node_count)
print("Time taken:", time.time()-s)


# In[444]:


#User defined Class for creating node object of decision tree
class Node:
    # Initializer
    def __init__(self, i, t, left, right, label, is_leaf):
        self.index = i
        self.threshold = t
        self.left = left
        self.right = right
        self.label = label
        self.is_leaf = is_leaf
    
    def set_leaf(self):
        self.is_leaf = True
    
    def unset_leaf(self):
        self.is_leaf = False
        
    def to_string(self):
        return str(self.index)+" "+str(self.threshold)+" "+str(self.left)+" "+str(self.right)

#Helper function to get height of a tree with root a    
def get_height(a):   
    if(a==None):
        return -1
    elif(a.is_leaf):
        return 0
    else:
        left_height = get_height(a.left)
        right_height = get_height(a.right)
        return 1 + max(left_height, right_height)

#Helper function to get count of internal(decision) nodes of a tree with root a    
def nodes_count(a):
    if(a==None):
        return 0
    elif(a.is_leaf):
        return 0
    else:
        return 1 + nodes_count(a.left) + nodes_count(a.right)

#Helper function to get count of all internal(decision) and external(leaves) nodes of a tree with root a    
def node_leaves_count(a):
    if(a==None):
        return 0
    elif(a.is_leaf):
        return 1
    else:
        return 1 + node_leaves_count(a.left) + node_leaves_count(a.right)


# In[167]:


#Helper function to get entropy of a variable with just two labels having their counts as parameters
def entropy(count_0, count_1):
    
    h_y = None
    
    if(count_0==0 or count_1==0):
        h_y = 0
    else:
        p_0 = count_0/(count_0 + count_1)
        p_1 = count_1/(count_0 + count_1)
        h_y = -(p_0*np.log2(p_0) + p_1*np.log2(p_1))
        
    return h_y

#To select the best attribute of a given dataset
#Returns the best attribute's index and the median value associated with it
def best_attribute(dataset, y, count_0, count_1):
    
    h_y = entropy(count_0, count_1)
    
    #Median value of attributes
    medians = np.median(dataset, axis = 0)
    non_zeros = np.count_nonzero(dataset, axis=0)
    
    best_index = -1
    best_ig = -float('inf')
    best_median = None
    
    total = count_0 + count_1
    
    #Iterating for each attribute
    for index, median in enumerate(medians):
        
        if(non_zeros[index] == 0):
            if(0>best_ig):
                best_index = index
                best_ig = 0
                best_median = median 
        
        index_neg = np.argwhere(dataset[:,index] <= median).flatten()
        index_pos = np.argwhere(dataset[:,index] > median).flatten()
        count_neg = index_neg.shape[0]
        count_pos = index_pos.shape[0]
        
        #if all the data belong to just one label, then information gain is 0 in that case
        if(count_neg==total or count_pos==total):
            if(0>best_ig):
                best_index = index
                best_ig = 0
                best_median = median
        else:
            
            y_neg = y[index_neg]
            y_pos = y[index_pos]
            
            count_1_neg = np.count_nonzero(y_neg)
            entropy_neg = (count_neg/total) * entropy(count_neg - count_1_neg, count_1_neg)
            
            count_1_pos = np.count_nonzero(y_pos)
            entropy_pos = (count_pos/total) * entropy(count_pos - count_1_pos, count_1_pos)
            
            ig = h_y - entropy_neg - entropy_pos
            
            #Selecting best attribute based on information gain
            if(ig>best_ig):
                best_index = index
                best_ig = ig
                best_median = median
        
    return best_index, best_median


# In[168]:


#Recursive function to grow the tree using ID3 algorithm
def grow_tree(dataset, y):
    
    count = y.shape[0]
    
    if(count==0):
        return None
    
    index_0 = np.argwhere(y==0)
    index_1 = np.argwhere(y==1)
    
    #Stopping criteria for growing tree, if all the data belong to just one label
    if(index_1.shape[0] == count):
        return Node(None, None, None, None, 1, True)
    elif(index_0.shape[0] == count):
        return Node(None, None, None, None, 0, True)
    else:
        
        label = -1
        if(index_1.shape[0] > index_0.shape[0]):
            label = 1
        else:
            label = 0
            
        index, threshold = best_attribute(dataset, y, index_0.shape[0], index_1.shape[0])
        
        index_neg = np.argwhere(dataset[:,index] <= threshold).flatten()
        index_pos = np.argwhere(dataset[:,index] > threshold).flatten()
        
        data_0, y_0 = dataset[index_neg][:], y[index_neg]
        data_1, y_1 = dataset[index_pos][:], y[index_pos]
        
        #Stopping criteria for growing tree
        #if all the data after splitting on best attribute belong to just one label
        if(y_1.shape[0] == 0 or y_0.shape[0] == 0):
            return Node(index, threshold, None, None, label, True)
        
        return Node(index, threshold, grow_tree(data_0, y_0), grow_tree(data_1, y_1), label, False)


# In[169]:


#Growing a tree
s = time.time()
m = grow_tree(x_train, y_train)
print("Time taken is:",time.time()-s)


# In[287]:


#Predicting label for a given row of data
def pred_row(root, row):
    n = root
    while(n!=None):
        if(n.is_leaf):
            return n.label
        else:
            i = n.index
            t = n.threshold
            if(row[i] <= t):
                n = n.left
            else:
                n = n.right
    return -1

#Calculating accuracy for a dataset for given decision tree
def accuracy(root, x, y):
    res = np.vectorize(lambda i: pred_row(root, x[i]))(np.arange(y.shape[0]))
    err = abs(y - res).sum()
    return 1 - err/y.shape[0]


# In[171]:


#Getting all the informations about the learned decision tree
s = time.time()
print("Accuracy over training set is: ", accuracy(m, x_train, y_train)*100,"%")
print("Accuracy over Validation set is: ", accuracy(m, x_val, y_val)*100,"%")
print("Accuracy over Test set is: ", accuracy(m, x_test, y_test)*100,"%")
print("Time taken to predict: ", time.time()-s)
print("Height of the fully grown tree is:", get_height(m))
#Internal(decision) nodes
print("Total number of nodes excluding leaves in fully grown tree is:", node_count(m))
#Internal(decision) plus external(leaf) nodes
print("Total number of nodes including leaves in fully grown tree is:", node_leaves_count(m))


# In[172]:


#A function that gets Validation, Test and Train accuracy while learning the tree
#As the nodes are added in the tree, this function is called to get the accuracy
def get_points():
    
    global n_nodes
    
    val_plot.append( accuracy(root, x_val, y_val))
    test_plot.append( accuracy(root, x_test, y_test))
    train_plot.append( accuracy(root, x_train, y_train))
    x_axis.append(n_nodes+1)
    
    print(n_nodes)


# In[173]:


#A different version of recursive algorithm to learn decision tree using ID3 algorithm
#This is done so that accuracy can also be calculated for datasets as we learn the tree
#Accuracy is calculated for every 10 nodes added in the tree

def grow_tree_v2(node, dataset, y):
    
    global n_nodes 
    
    count = y.shape[0]
    
    if(count==0):
        node = None
        return
    
    index_0 = np.argwhere(y==0)
    index_1 = np.argwhere(y==1)
    
    #Stopping criteria for growing tree, if all the data belong to just one label
    if(index_1.shape[0] == count):
        node.label = 1
        node.is_leaf = True
        
        if(n_nodes%10 == 0):
            get_points()
        n_nodes += 1
        
        return
    
    #Stopping criteria for growing tree, if all the data belong to just one label  
    elif(index_0.shape[0] == count):
        node.label = 0
        node.is_leaf = True
        
        if(n_nodes%10 == 0):
            get_points()
        n_nodes += 1
        
        return
        
    else:
        
        label = -1
        if(index_1.shape[0] >= index_0.shape[0]):
            label = 1
        else:
            label = 0
        
        node.label = label
        node.is_leaf = True
        
        #For every 10 nodes, the function is called to get the accuracy for different datasets
        if(n_nodes%10 == 0):
            get_points()
        n_nodes += 1
        
        node.is_leaf = False
        
        index, threshold = best_attribute(dataset, y, index_0.shape[0], index_1.shape[0])
        
        index_neg = np.argwhere(dataset[:,index] <= threshold).flatten()
        index_pos = np.argwhere(dataset[:,index] > threshold).flatten()
        
        data_0, y_0 = dataset[index_neg][:], y[index_neg]
        data_1, y_1 = dataset[index_pos][:], y[index_pos]
        
        node.index = index
        node.threshold = threshold
        
        #Stopping criteria for growing tree
        #if all the data after splitting on best attribute belong to just one label
        if(y_1.shape[0] == 0 or y_0.shape[0] == 0):
            node.is_leaf = True
            return
        
        label_r = np.count_nonzero(y_1)
        if((y_1.shape[0]-label_r) > label_r):
            label_r = 0
        else:
            label_r = 1
            
        node.left = Node(None, None, None, None, -1, False)
        node.right = Node(None, None, None, None, label_r, True)
        grow_tree_v2(node.left, data_0, y_0)
        
        node.right = Node(None, None, None, None, -1, False)
        grow_tree_v2(node.right, data_1, y_1)
        
        return


# In[174]:


#Root is made global
root = Node(None, None, None, None, -1, False)
#Different lists for storing accuracy and node counts as the tree grows
val_plot = []
train_plot = []
test_plot = []
x_axis = []
n_nodes = 0

#Learning the tree using the second version of grow_tree function
s=time.time()
grow_tree_v2(root, x_train, y_train)
print("Time taken: ", time.time()-s)


# In[176]:


#Getting all the informations about the learned decision tree
#It comes out to be same as the previous version
s = time.time()
print("Accuracy over training set is: ", accuracy(root, x_train, y_train)*100,"%")
print("Accuracy over Validation set is: ", accuracy(root, x_val, y_val)*100,"%")
print("Accuracy over Test set is: ", accuracy(root, x_test, y_test)*100,"%")
print("Height of the fully grown tree is:", get_height(root))
print("Total number of nodes excluding leaves in fully grown tree is:", node_count(root))
print("Total number of nodes including leaves in fully grown tree is:", node_leaves_count(root))


# In[179]:


#Saving the info in file for future reference so that time is saved
np.savetxt('plots.txt', (x_axis, val_plot, test_plot, train_plot), delimiter=' ')


# In[441]:


#Loading the saved info about accuracy as decision tree grows
#This part is invoked only if required
p = np.loadtxt('plots.txt')
x_axis = p[0]
val_plot = p[1]
test_plot = p[2]
train_plot = p[3]


# In[443]:


#Plotting the required results
plt.figure()
lw=2
plt.plot(x_axis, val_plot, color='darkorange', lw=lw, label='Validation Accuracy')
plt.plot(x_axis, test_plot, color='navy', lw=lw, linestyle='--', label='Test Accuracy')
plt.plot(x_axis, train_plot, color='darkgreen', lw=lw, linestyle='dotted', label='Train Accuracy')
plt.xlim([0, 20500])
plt.ylim([0.5, 0.95])
plt.ylabel('Accuracy')
plt.xlabel('Number of Nodes')
plt.title('Accuracy with increasing number of nodes')
plt.legend(loc="lower right")
plt.show()


# In[367]:


#Function to get BFS list of a tree with given root, only the internal(decision) nodes are counted
def bfs_nodes(root):
    
    check = [root]
    bfs = []
    
    while(len(check) > 0):
        
        n = check.pop(0)
        
        if(n.is_leaf or n==None):
            continue        
        
        check.append(n.left)
        check.append(n.right)
        bfs.append(n)
        
    return bfs


# In[376]:


#Function to get BFS list of a tree with given root, all internal(decision) as well as external nodes are counted
def bfs_all(root):
    
    check = [root]
    bfs = []
    
    while(len(check) > 0):
        
        n = check.pop(0)
        bfs.append(n)
        
        if(n.is_leaf or n==None):
            continue        
        
        check.append(n.left)
        check.append(n.right)

    return bfs


# In[409]:


#Function to get BFS list of a tree with given root, only the external(leaf) nodes are counted
def bfs_leaf(root):
    
    check = [root]
    bfs = []
    
    while(len(check) > 0):
        
        n = check.pop(0)
        
        if(n.is_leaf or n==None):
            bfs.append(n)
            continue        
        
        check.append(n.left)
        check.append(n.right)
        
    return bfs


# In[ ]:


#Growing the tree with 1st version, takes less time, accuracy is not required.
s = time.time()
root = grow_tree(x_train, y_train)
print("Time taken is:",time.time()-s)


# In[ ]:


l = bfs(root)


# In[ ]:


for i in l:
    i.is_leaf = False


# In[410]:


node_leaves_count(root), get_height(root)


# In[379]:


#As the validation data is predicted, this part of code stores the count of each label 
#for the subset of dataset that reaches each node.
#Helps in calculating number of validation datasets that are being misclassified at each node

d = []
d.append({})
d.append({})

for i in range(len(x_val)):
    x = x_val[i]
    y = int(y_val[i])
    
    n = root
    while(n!=None):
        
        if(n in d[y]):
            d[y][n] = d[y].get(n) + 1
        else:
            d[y][n] = 1
            
            
        if(n.is_leaf):
            break
        else:
            i = n.index
            t = n.threshold
            if(x[i] <= t):
                n = n.left
            else:
                n = n.right
    


# In[380]:


#A dictionary that stores number of validation data set that are misclassified 
#at each node if that node was a leaf
mis = {}
all_nodes = bfs_all(root)


# In[381]:


#For each node in tree store the misclassified datas at each node

#This is done so that when we are pruning the tree we don't have to call accuracy function(saves time)
for n in all_nodes:
    label = n.label
    
    if(n not in d[1-label] and n not in d[label]):
        mis[n] = None
    
    if(n in d[1-label]):
        mis[n] = d[1-label][n]
    else:
        mis[n] = 0


# In[411]:


#A function to post prune the tree based on pruning node that increased maximum accuracy

#We can directly get the number of misclassifications of validation dataset if that node was a leaf, as
#compared to number of misclassification if that whole subtree(with that node as root) was being used
#If the node gives less misclassification than the subtree, then it's better to prune the node.
#We can get the node that gives maximum decrease in number of misclassification and prune that one

#It saves al lot of time and speeds up pruning process by an enormous factor.

def prune(l, e):
    
    prune_node = None
    curr_e = 0
    
    while(len(bfs_list) > 0):
        
        node = bfs_list.pop()
        
        #Misclassification if that node was a leaf
        p_err = e[node]
        c_err = 0
        
        #All the leaves of that subtree
        child = bfs_leaf(node)
        
        #Sum of misclassification by that entire subtree
        for c in child:
            c_err += e[c]
        
        #Getting the node with maximum decrease in misclassifications after pruning
        if((c_err - p_err) > curr_e):
            curr_e = c_err - p_err
            prune_node = node
    
    if(prune_node == None):
        return False
    else:
        prune_node.is_leaf = True
        return True


# In[ ]:


# def max_prune(bfs_list):
    
#     curr_acc = accuracy_new(root, x_val, y_val)
#     prune_node = None
    
#     while(len(bfs_list) > 0):
        
#         node = bfs_list.pop()
        
#         node.is_leaf = True
#         acc = accuracy_new(root, x_val, y_val)
        
#         if(acc > curr_acc):
#             curr_acc = acc
#             prune_node = node
        
#         node.is_leaf = False
    
#     if(prune_node == None):
#         return False
#     else:
#         prune_node.is_leaf = True
#         return True


# In[412]:


# Part of the code that prunes the tree as long as there is no increase in accuracy over validation set

s = time.time()

acc_val = [accuracy(root, x_val, y_val)]
acc_test = [accuracy(root, x_test, y_test)]
acc_train = [accuracy(root, x_train, y_train)]
node_count = [node_leaves_count(root)]

print(acc_val, acc_test, acc_train, node_count)

prune_count = 0

while(True):
    
    s1 = time.time()
    print("Going for prune_count: ", prune_count+1)
    
    #Getting all the internal(decision) nodes to be considered for pruning
    bfs_list = bfs_nodes(root)
    
    #Getting result if pruning is possible or not
    res = prune(bfs_list, mis)
    
    print("Done for prune_count: ", prune_count+1, "in time:", time.time()-s1)
    
    if(res):
        
        acc_val.append(accuracy(root, x_val, y_val))
        acc_test.append(accuracy(root, x_test, y_test))
        acc_train.append(accuracy(root, x_train, y_train))
        
        count = node_leaves_count(root)
        node_count.append(count)
        
        prune_count += 1
        print("Node Count: ", count)
        
        continue
        
    break
    
print("Pruning done, time taken : ", time.time()-s)


# In[413]:


#Number of external(leaf) + internal(decision) nodes in pruned tree
# number of internal(decision) nodes in pruned tree
node_leaves_count(root), get_height(root)


# In[428]:


#Saving info for future reference so that don't have to do gain
np.savetxt('prune.txt', (node_count, acc_val, acc_test, acc_train), delimiter=' ')


# In[432]:


#Plotting the desired result
plt.figure()
lw=2
plt.plot(node_count, acc_val, color='darkorange', lw=lw, label='Validation Accuracy')
plt.plot(node_count, acc_test, color='navy', lw=lw, linestyle='--', label='Test Accuracy')
plt.plot(node_count, acc_train, color='darkgreen', lw=lw, linestyle='dotted', label='Train Accuracy')
plt.xlim([21000, 4000])
plt.ylim([0.75, 0.95])
plt.ylabel('Accuracy')
plt.xlabel('Number of Nodes')
plt.title('Accuracy with decreasing number of nodes after successive pruning')
plt.legend(loc="upper right")
plt.show()


# In[423]:


#Getting accuracy after pruning is done
accuracy(root, x_val, y_val), accuracy(root,x_test,y_test), accuracy(root, x_train, y_train)


# In[426]:


#Getting count of decision nodes in pruned tree
nodes_count(root)


# In[435]:


s = time.time()
m = grow_tree(x_train, y_train)
print("Time taken is:",time.time()-s)


# In[436]:


#getting count of total nodes in left and right subtree of root
node_leaves_count(m.left), node_leaves_count(m.right)

