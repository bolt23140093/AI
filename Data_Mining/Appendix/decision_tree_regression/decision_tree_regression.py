# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 22:43:05 2023

@author: joseph@艾鍗學院

 Decision Tree Regressor

"""

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

# Generate a synthetic dataset
# Let's generate some data that roughly follows the equation y = sin(x)
rng = np.random.default_rng(seed=42)
#generate data following unifrom distribution uniform(a,b)
X = np.sort(5 * rng.uniform(0, 1, 80))[:, np.newaxis] #(80,)-->(80,1)
y = np.sin(X).ravel() + rng.normal(0, 0.1, X.shape[0]) #(80,1)-->(80,)

print(X.shape,y.shape)


# Train a DecisionTreeRegressor
tree = DecisionTreeRegressor(max_depth=2)
tree.fit(X, y)

# Predict
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
y_pred = tree.predict(X_test)



print(X_test.shape)

n_nodes = tree.tree_.node_count
feature = tree.tree_.feature
threshold = tree.tree_.threshold

print(n_nodes)

# Iterate through all nodes
for i in range(n_nodes):
    if feature[i] != -2:  # check if not a leaf node
        print(f"Node {i} splits on feature index {feature[i]} at threshold {threshold[i]}")


# Plot the results
plt.figure()
plt.scatter(X, y, color="darkorange", label="data")
plt.plot(X_test, y_pred, color="blue", label="prediction", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()
