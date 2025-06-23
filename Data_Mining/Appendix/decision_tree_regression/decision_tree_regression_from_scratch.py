# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 22:51:27 2023

@author: joseph@艾鍗學院
"""

import numpy as np
import matplotlib.pyplot as plt

# Define the Node and Leaf structures
class TreeNode:
    def __init__(self, feature_index, threshold, left, right, X_left,X_right):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.left_data= X_left
        self.right_data= X_right

class LeafNode:
    def __init__(self, value):
        self.value = value

# Utility function to split data

def split_data(X, y, feature_index, threshold):
    left_indices = np.where(X[:, feature_index] <= threshold)
    right_indices = np.where(X[:, feature_index] > threshold)
     
    return X[left_indices], y[left_indices], X[right_indices], y[right_indices]

def split_datav2(X, y, feature_index, threshold):
    left_mask = X[:, feature_index] <= threshold
    right_mask = ~left_mask
    return X[left_mask], y[left_mask], X[right_mask], y[right_mask]



# Cost function (Mean Squared Error)
def mse(y):
    if len(y) == 0: return 0
    mean_y = np.mean(y)
    return np.mean((y - mean_y) ** 2)

# Function to find best split
def find_best_split(X, y):
    best_feature, best_threshold, best_cost, best_left, best_right = \
        None, None, float('inf'), None, None
    n_samples, n_features = X.shape

    for feature_index in range(n_features):
        thresholds = np.unique(X[:, feature_index])
        for threshold in thresholds:
            X_left, y_left, X_right, y_right = split_data(X, y, feature_index, threshold)
            current_cost = mse(y_left) + mse(y_right)
            
            if current_cost < best_cost:
                best_feature = feature_index
                best_threshold = threshold
                best_cost = current_cost
                best_left = (X_left, y_left)
                best_right = (X_right, y_right)

    return best_feature, best_threshold, best_left, best_right


# Recursive function to build the tree
def build_tree(X, y, depth=0, max_depth=None):
    n_samples, n_features = X.shape
    
    # Stopping conditions
    if n_samples == 0:
        return None
    if max_depth and depth == max_depth:
        return LeafNode(np.mean(y))
    
    feature, threshold, (X_left, y_left), (X_right, y_right) = find_best_split(X, y)

    left_tree = build_tree(X_left, y_left, depth + 1, max_depth)
    right_tree = build_tree(X_right, y_right, depth + 1, max_depth)

    
    return TreeNode(feature, threshold, left_tree, right_tree,X_left,X_right)

# Prediction function
def predict_tree(sample, tree):
    #isinstance(Obj,Class) returns True if the specified object is of the specified type/class
    
   if tree is None : 
       return 
    
   if isinstance(tree, LeafNode):
        return tree.value
    #print(type(tree),tree.feature_index,tree.threshold)
   if sample[tree.feature_index] <= tree.threshold:
        return predict_tree(sample, tree.left)
   else:
        return predict_tree(sample, tree.right)

# Traverse Tree function
def traverse_tree(sample,tree,depth=1):
    #isinstance(Obj,Class) returns True if the specified object is of the specified type/class
    if tree is None : 
        return 

    if isinstance(tree, LeafNode): #isinstance(5,int)=>True
      
        print('-'*depth,tree.value)
        
    
    else :
        
         traverse_tree(sample,tree.left,depth+1)
         print('left tree:',tree.left_data)
         print('right tree:',tree.right_data)
         traverse_tree(sample,tree.right,depth+1)
         
    



# Generate a synthetic dataset
rng = np.random.default_rng(seed=42)
X = np.sort(5 * rng.uniform(0, 1, 10))[:, np.newaxis]  # matrix (n,1)
#X = np.sort(np.random.randint(0, 5, 10))[:, np.newaxis]  # matrix (n,1)
print(X.shape)
y = np.sin(X).ravel() + rng.normal(0, 0.1, X.shape[0]) # vector (n,)  

# Train a regression tree from scratch
tree = build_tree(X, y,max_depth=3)

# Predict
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
print(X_test.shape)

traverse_tree(X,tree)

y_pred = [predict_tree(sample, tree) for sample in X_test]



# Plot the results
plt.figure()
plt.scatter(X, y, color="darkorange", label="data")
plt.plot(X_test, y_pred, color="cornflowerblue", label="prediction", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression from Scratch")
plt.legend()
plt.show()
