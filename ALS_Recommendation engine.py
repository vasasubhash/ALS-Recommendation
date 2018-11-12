# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 12:38:55 2018

@author: Subhash
"""

import pandas as pd
import scipy.sparse as sparse
import implicit

# Loading and preparing the data
raw_data = pd.read_table('datapath/usersha1-artmbid-artname-plays.tsv',encoding = 'latin')
raw_data = raw_data.drop(raw_data.columns[1], axis=1)
raw_data.columns = ['user', 'artist', 'plays']

# Drop NaN columns
data = raw_data.dropna()
data = data.copy()

# Create a numeric user_id and artist_id column
data['user'] = data['user'].astype("category")
data['artist'] = data['artist'].astype("category")
data['user_id'] = data['user'].cat.codes
data['artist_id'] = data['artist'].cat.codes

# The implicit library expects data as a item-user matrix so we
# create two matricies, one for fitting the model (item-user) 
# and one for recommendations (user-item)
sparse_item_user = sparse.csr_matrix((data['plays'].astype(float), (data['artist_id'], data['user_id'])))
sparse_user_item = sparse.csr_matrix((data['plays'].astype(float), (data['user_id'], data['artist_id'])))

# Initialize the als model and fit it using the sparse item-user matrix
model = implicit.als.AlternatingLeastSquares(factors=20, regularization=0.1, iterations=20)

# Calculate the confidence by multiplying it by our alpha value.
alpha_val = 15
data_conf = (sparse_item_user * alpha_val).astype('double')

#Fit the model
model.fit(data_conf)


#---------------------
# FIND SIMILAR ITEMS
#---------------------

# Find the 10 most similar to anemo
item_id = 25461  #anemo
n_similar = 10

# Use implicit to get similar items.
similar = model.similar_items(item_id, n_similar)

# Print the names of our most similar artists
for item in similar:
    idx, score = item
    print (data.artist.loc[data.artist_id == idx].iloc[0])

    
#------------------------------
# CREATE USER RECOMMENDATIONS
#------------------------------

# Create recommendations for user with id 37779
user_id = 37779

# Use the implicit recommender.
recommended = model.recommend(user_id, sparse_user_item)

artists = []
scores = []

# Get artist names from ids
for item in recommended:
    idx, score = item
    artists.append(data.artist.loc[data.artist_id == idx].iloc[0])
    scores.append(score)

# Create a dataframe of artist names and scores
recommendations = pd.DataFrame({'artist': artists, 'score': scores})

print (recommendations)

