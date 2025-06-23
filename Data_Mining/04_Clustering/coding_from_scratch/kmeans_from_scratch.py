# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 09:21:04 2024

@author: joseph@艾鍗學院 www.ittraining.com.tw
"""

import numpy as np

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

def k_means(data, k, max_iters=100):
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]
    
    for _ in range(max_iters):
        labels = np.argmin(np.array([[euclidean_distance(data[i], centroids[j]) for j in range(k)] for i in range(data.shape[0])]), axis=1)
        new_centroids = np.array([data[labels == j].mean(axis=0) for j in range(k)])
        
        if np.all(centroids == new_centroids):
            break
        
        centroids = new_centroids
    
    return centroids, labels

# Generate some sample data
data = np.array([[1, 2],
                 [1, 3],
                 [1.5, 1.8],
                 [8, 8],
                 [8, 7.6],
                 [9, 11]])

# Define the number of clusters (K)
k = 2

# Apply K-means clustering
centroids, labels = k_means(data, k)

print("Final centroids:")
print(centroids)
print("Cluster labels:")
print(labels)
