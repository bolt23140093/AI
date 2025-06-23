# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 12:51:53 2024

@author: joseph@艾鍗學院 www.ittraining.com.tw
"""

import numpy as np

# Function to calculate cosine similarity
def cosine_similarity(vecA, vecB):
    dot_product = np.dot(vecA, vecB)
    normA = np.linalg.norm(vecA)
    normB = np.linalg.norm(vecB)
    return dot_product / (normA * normB)

# Vectors
a = np.array([ .3, .3, .1])
b = np.array([ 0,   0, .10])
c = np.array([0, 0, 2])
d = np.array([3, 3, 1])

#c = np.array([.1,  .1,  .1])

#c = np.array([3, 3, 3])

# Cosine similarities.
cos_sim_aa = cosine_similarity(a, a)
cos_sim_ab = cosine_similarity(a, b)
cos_sim_ac = cosine_similarity(a, c)
cos_sim_ad = cosine_similarity(a, d)
# Display the results.
print("Cosine Similarity between a and a:", cos_sim_aa)
print("Cosine Similarity between a and b:", cos_sim_ab)
print("Cosine Similarity between a and c:", cos_sim_ac)
print("Cosine Similarity between a and d:", cos_sim_ad)

