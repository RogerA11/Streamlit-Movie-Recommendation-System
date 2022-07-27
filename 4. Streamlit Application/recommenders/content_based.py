"""

    Content-based filtering for item recommendation.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: You are required to extend this baseline algorithm to enable more
    efficient and accurate computation of recommendations.

    !! You must not change the name and signature (arguments) of the
    prediction function, `content_model` !!

    You must however change its contents (i.e. add your own content-based
    filtering algorithm), as well as altering/adding any other functions
    as part of your improvement.

    ---------------------------------------------------------------------

    Description: Provided within this file is a baseline content-based
    filtering algorithm for rating predictions on Movie merged_df.

"""

# Script dependencies
from heapq import merge
import os
import pandas as pd
import numpy as np
import functools as ft
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# import data
movies = pd.read_csv('./resources/data/movies.csv', sep = ',')
imdb = pd.read_csv('./resources/data/imdb_data.csv')
movies = movies.dropna()

def data_preprocessing(subset_size):
    """Prepare merged_df for use within Content filtering algorithm.

    Parameters
    ----------
    subset_size : int
        Number of movies to use within the algorithm.

    Returns
    -------
    Pandas Dataframe
        Subset of movies selected for content-based filtering.

    """
    # create merged_df
    # dfs = [movies, imdb]
    # merged_df = ft.reduce(lambda left, right: pd.merge(left, right, on='movieId'), dfs)
    merged_df = movies.merge(imdb, how='left', on='movieId')

    # fill in null values 
    merged_df['title_cast'] = merged_df['title_cast'].fillna(merged_df['title_cast'].mode()[0])
    merged_df['plot_keywords'] = merged_df['plot_keywords'].fillna(merged_df['plot_keywords'].mode()[0])
    merged_df['director'] = merged_df['director'].fillna(merged_df['director'].mode()[0])

    # data preprocessing
    # clean title feature and create a new feature called year
    merged_df['title'] = merged_df['title'].str.split('(')
    merged_df['title'] = merged_df['title'].str[0]
    merged_df['title'] = merged_df['title'].str.rstrip()
    #merged_df['year'] = merged_df['title'].str[1]
    #merged_df['year'] = merged_df['year'].str.replace(')','',regex=True)

    # clean genres, title_cast and plot_keywords features
    merged_df['genres'] = merged_df['genres'].str.split('|')
    merged_df['title_cast'] = merged_df['title_cast'].str.split('|')
    merged_df['plot_keywords'] = merged_df['plot_keywords'].str.split('|')

    # prepare director feature
    merged_df['director'] = merged_df['director'].astype('str').apply(lambda x: str.lower(x.replace(" ", "")))
    merged_df['director'] = merged_df['director'].apply(lambda x: [x, x, x])

    # create input feature by merging various features
    input = merged_df['plot_keywords'] + merged_df['title_cast'] + merged_df['director'] + merged_df['genres']
    merged_df['input'] = input
    merged_df = merged_df.dropna(subset=['input'])
    merged_df['input'] = merged_df['input'].apply(lambda x: ' '.join(x))
    
    # create movies_subset variable
    movies_subset = merged_df.iloc[:subset_size, :]
    
    return movies_subset
    
# !! DO NOT CHANGE THIS FUNCTION SIGNATURE !!
# You are, however, encouraged to change its content.  
def content_model(movie_list,top_n=10):
    """Performs Content filtering based upon a list of movies supplied
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
    # Initializing the empty list of recommended movies
    recommended_movies = []
    merged_df = data_preprocessing(27000)

    # Instantiating and generating the count matrix
    count_vec = CountVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
    count_matrix = count_vec.fit_transform(merged_df['input'])
    indices = pd.Series(merged_df['title'])
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    
    # Getting the index of the movie that matches the title
    idx_1 = indices[indices == movie_list[0]].index[0]
    idx_2 = indices[indices == movie_list[1]].index[0]
    idx_3 = indices[indices == movie_list[2]].index[0]
    
    # Creating a Series with the similarity scores in descending order
    rank_1 = cosine_sim[idx_1]
    rank_2 = cosine_sim[idx_2]
    rank_3 = cosine_sim[idx_3]
    
    # Calculating the scores
    score_series_1 = pd.Series(rank_1).sort_values(ascending = False)
    score_series_2 = pd.Series(rank_2).sort_values(ascending = False)
    score_series_3 = pd.Series(rank_3).sort_values(ascending = False)
    
    # Getting the indexes of the 10 most similar movies
    listings = score_series_1.append(score_series_1).append(score_series_3).sort_values(ascending = False)

    # Store movie names
    recommended_movies = []
    
    # Appending the names of movies
    top_50_indexes = list(listings.iloc[1:50].index)
    
    # Removing chosen movies
    top_indexes = np.setdiff1d(top_50_indexes,[idx_1,idx_2,idx_3])
    for i in top_indexes[:top_n]:
        recommended_movies.append(list(movies['title'])[i])

    return recommended_movies
