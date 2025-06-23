# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 16:29:52 2023
@author: joseph@艾鍗學院
"""

import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

# Sample movie ratings matrix (rows: users, columns: movies)
# Each row represents a user's ratings for different movies (rating scale: 1-5)
ratings = np.array([
    [5, 4, 0, 0, 1, 0],
    [0, 0, 5, 4, 0, 2],
    [2, 0, 0, 0, 5, 4],
    [0, 2, 4, 0, 0, 5]
])

# Transpose the ratings matrix to get movies as rows and users as columns
movie_ratings = ratings.T
# Calculate cosine similarity between movies
movie_similarity = cosine_similarity(movie_ratings)
# Choose a user for recommendation
user_id = 1
user_ratings = ratings[user_id]

# Find movies rated by the chosen user
rated_movies = np.where(user_ratings > 0)[0]
unwatched_movies = np.where(user_ratings == 0)[0]

# Calculate recommendation scores for unrated movies
recommendations = np.zeros(ratings.shape[1])
for i in unwatched_movies:
    for j in rated_movies:
        recommendations[i] +=  movie_similarity[i, j] * user_ratings[j]
    
# Sort recommendations and suggest top movies
top_recommendations = np.argsort(recommendations)[::-1]

print("Top recommended movies:")
for i in range(3):
    print(f"Movie {top_recommendations[i]}")
