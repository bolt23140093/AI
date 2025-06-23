# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 17:20:46 2023

@author: joseph@艾鍗學院

User-based Collaborative Filtering

"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Sample movie ratings matrix (rows: users, columns: movies)
# Each row represents a user's ratings for different movies (rating scale: 1-5)
ratings = np.array([
    [5, 4, 0, 0, 1, 0],
    [0, 0, 5, 4, 0, 2],   
    [2, 0, 0, 0, 5, 4],
    [0, 2, 4, 2, 0, 5]
])

# Calculate cosine similarity between users
user_similarity = cosine_similarity(ratings)

# Choose a user for recommendation
user_id = 1
user_ratings = ratings[user_id]

# Find most similar user(s) for recommendation
most_similar_users = np.argsort(user_similarity[user_id])[::-1] #由後到前
print('most_similar_users:',most_similar_users)
most_similar_users = most_similar_users[1:]  # Exclude the user itself (it's always at zero position)

# Find movies not rated by the chosen user
unrated_movies = np.where(user_ratings == 0)[0]

# Suggest movies based on similar users' ratings
recommendations_score = np.zeros(ratings.shape[1])

for i in unrated_movies:
    for j in most_similar_users:
        #if ratings[similar_user, movie] > 0:
            recommendations_score[i] += user_similarity[user_id, j]*ratings[j, i]

# Sort recommendations and suggest top movies
top_recommendations = np.argsort(recommendations_score)[::-1]

print("Top recommended movies:")
for i in range(3):
    print(f"Movie {top_recommendations[i]}")
