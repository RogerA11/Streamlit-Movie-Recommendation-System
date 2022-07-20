"""

    Collaborative-based filtering for item recommendation.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: You are required to extend this baseline algorithm to enable more
    efficient and accurate computation of recommendations.

    !! You must not change the name and signature (arguments) of the
    prediction function, `collab_model` !!

    You must however change its contents (i.e. add your own collaborative
    filtering algorithm), as well as altering/adding any other functions
    as part of your improvement.

    ---------------------------------------------------------------------

    Description: Provided within this file is a baseline collaborative
    filtering algorithm for rating predictions on Movie data.

"""

# Script dependencies
import pandas as pd
import numpy as np
import scipy as sp
import operator
import pickle
import copy
from surprise import Reader, Dataset
from surprise import SVD, NormalPredictor, BaselineOnly, KNNBasic, NMF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Custom Libraries
from utils.data_loader import load_movie_titles

# Importing train and test datasets
ratings_df = pd.read_csv('resources/data/ratings.csv', usecols=['userId', 'movieId', 'rating'])
movies_df = pd.read_csv('resources/data/movies.csv',sep = ',')
# ratings_df.drop(['timestamp'], axis=1,inplace=True)

# We make use of an SVD model trained on a subset of the MovieLens 10k dataset.
model=pickle.load(open('resources/models/SVD_01.pkl', 'rb'))

# Create a subset of movies based on selectable options to improve performance
title_list = load_movie_titles('resources/data/movies.csv')
movie_list_1 = title_list[14930:15200]
movie_list_2 = title_list[25055:25255]
movie_list_3 = title_list[21100:21200]
combined_movies_list = movie_list_1 + movie_list_2 + movie_list_3
movie_options = movies_df[movies_df['title'].isin(combined_movies_list)]

# Create a subset based of the option above
def ratings_subset(movie_options, ratings_df):
    """Create a subset based on the movie options

    Parameters
    ----------
    movie_options : list
        A subset of the movies_df.
    ratings_df : DataFrame
        Rating DataFrame to subset

    Returns
    -------
    DataFrame
        A subset of the ratings DataFrame.

    """
    ratings_ids = ratings_df[ratings_df['movieId'].isin(movie_options['movieId'].tolist())]
    ratings_subset = ratings_df[ratings_df['userId'].isin(ratings_ids['userId'].tolist())]
    return ratings_subset

# Return a subset of ratings
listed_ratings = ratings_subset(movie_options, ratings_df)


def prediction_item(item_id):
    """Map a given favourite movie to users within the
       MovieLens dataset with the same preference.

    Parameters
    ----------
    item_id : int
        A MovieLens Movie ID.

    Returns
    -------
    list
        User IDs of users with similar high ratings for the given movie.

    """
    # Data preprosessing
    reader = Reader(rating_scale=(0, 5))
    load_df = Dataset.load_from_df(listed_ratings, reader)
    a_train = load_df.build_full_trainset()

    predictions = []
    for ui in a_train.all_users():
        predictions.append(model.predict(iid=item_id, uid=ui, verbose = False))
    return predictions

def pred_movies(movie_list):
    """Maps the given favourite movies selected within the app to corresponding
    users within the MovieLens dataset.

    Parameters
    ----------
    movie_list : list
        Three favourite movies selected by the app user.

    Returns
    -------
    list
        User-ID's of users with similar high ratings for each movie.

    """
    # Store the id of users
    id_store=[]
    # For each movie selected by a user of the app,
    # predict a corresponding user within the dataset with the highest rating
    for i in movie_list:
        predictions = prediction_item(item_id = i)
        predictions.sort(key=lambda x: x.est, reverse=True)
        # Take the top 50 user id's from each movie with highest rankings
        for pred in predictions[:50]:
            id_store.append(pred.uid)
    # Return a list of user id's
    return id_store

def collab_model(movie_list,top_n):
    """Performs Collaborative filtering based upon a list of movies supplied
       by the app user.

    Parameters
    ----------
    movie_list : list (str)
        Favorite movies chosen by the app user.
    top_n : type
        Number of top recommendations to return to the user.

    Returns
    -------
    list (str)
        Titles of the top-n movie recommendations to the user.

    """

    # get movie ids for movie_list
    movie_ids = []
    for movie in movie_list:
        movie_ids.append(int(movie_options['movieId'][movie_options['title']==movie]))

    # Create list of users which would rate these movies highly
    user_ids = pred_movies(movie_list)

    # Create dataframe of all the movies that these users have rated
    df_init_users = listed_ratings[listed_ratings['userId'].isin(user_ids)]

    # Add new user with ratings to userlist
    new_rating_1 = {'userId':200000, 'movieId':movie_ids[0], 'rating':5.0}
    new_rating_2 = {'userId':200000, 'movieId':movie_ids[1], 'rating':5.0}
    new_rating_3 = {'userId':200000, 'movieId':movie_ids[2], 'rating':4.5}
    df_init_users = df_init_users.append([new_rating_1,new_rating_2,new_rating_3], ignore_index=True)

    # Creating Util matrix,replace NANs and transpose
    util_matrix = pd.pivot_table(df_init_users,values='rating',columns='movieId',index='userId')
    util_matrix_norm = util_matrix.apply(lambda x: (x-np.mean(x))/(np.max(x)-np.min(x)), axis=1)
    util_matrix_norm = util_matrix_norm.fillna(0)
    util_matrix_norm = util_matrix_norm.T
    util_matrix_norm = util_matrix_norm.loc[:, (util_matrix_norm != 0).any(axis=0)]

    # Save the utility matrix in scipy's sparse matrix format
    util_matrix_sparse = sp.sparse.csr_matrix(util_matrix_norm.values)

    # Compute the similarity matrix using the cosine similarity metric
    user_similarity = cosine_similarity(util_matrix_sparse.T)

    # Save the matrix as a dataframe to allow for easier indexing
    user_sim_df = pd.DataFrame(user_similarity,
                               index = util_matrix_norm.columns,
                               columns = util_matrix_norm.columns)

    user = 200000
    k=50
    # Gather the k users which are most similar to the reference user
    sim_users = user_sim_df.sort_values(by=user, ascending=False).index[1:k+1]
    favorite_user_items = [] # <-- List of highest rated items gathered from the k users
    most_common_favorites = {} # <-- Dictionary of highest rated items in common for the k users

    for i in sim_users:
        # Maximum rating given by the current user to an item
        max_score = util_matrix_norm.loc[:, i].max()
        # Save the names of items maximally rated by the current user
        favorite_user_items.append(util_matrix_norm[util_matrix_norm.loc[:, i]==max_score].index.tolist())

    # Loop over each user's favorite items and tally which ones are
    # most popular overall.
    for item_collection in range(len(favorite_user_items)):
        for item in favorite_user_items[item_collection]:
            if item in most_common_favorites:
                most_common_favorites[item] += 1
            else:
                most_common_favorites[item] = 1
    # Sort the overall most popular items and return the top-N instances
    sorted_list = sorted(most_common_favorites.items(), key=operator.itemgetter(1), reverse=True)[:top_n+1]
    top_N = [x[0] for x in sorted_list]

    # Return Movie Names
    recommendations = []
    for movieid in top_N:
        recommendations.append(movies_df[movies_df['movieId']==movieid]['title'].tolist())
    recommendations = [item for sublist in recommendations for item in sublist]
    recommendations = [x for x in recommendations if x not in movie_list]
    recommendations[:top_n]
    return recommendations
