"""

    Streamlit webserver-based Recommender Engine.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: !! Do not remove/modify the code delimited by dashes !!

    This application is intended to be partly marked in an automated manner.
    Altering delimited code may result in a mark of 0.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend certain aspects of this script
    and its dependencies as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import streamlit.components.v1 as components

# Data handling dependencies
import pandas as pd
import numpy as np

# Custom Libraries
from utils.data_loader import load_movie_titles, markdown_reader, local_css
from recommenders.collaborative_based import collab_model
from recommenders.content_based import content_model

# Data Loading
title_list = load_movie_titles('resources/data/movies.csv')

# Load pages
team = markdown_reader("resources/pages/meet_the_team.html")

# App declaration
def main():

    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.
    page_options = ["Recommender System", "Solution Overview","Exploratory Data Analysis","About Us"]

    # -------------------------------------------------------------------
    # ----------- !! THIS CODE MUST NOT BE ALTERED !! -------------------
    # -------------------------------------------------------------------
    page_selection = st.sidebar.selectbox("Menu", page_options)
    if page_selection == "Recommender System":
        # Header contents
        st.write('# Movie Recommender Engine')
        st.write('### EXPLORE Data Science Academy Unsupervised Predict')
        st.image('resources/imgs/Image_header.png',use_column_width=True)
        # Recommender System algorithm selection
        sys = st.radio("Select an algorithm",
                       ('Content Based Filtering',
                        'Collaborative Based Filtering'))

        # User-based preferences
        st.write('### Enter Your Three Favorite Movies')
        movie_1 = st.selectbox('Fisrt Option',title_list[14930:15200])
        movie_2 = st.selectbox('Second Option',title_list[25055:25255])
        movie_3 = st.selectbox('Third Option',title_list[21100:21200])
        fav_movies = [movie_1,movie_2,movie_3]

        # Perform top-10 movie recommendation generation
        if sys == 'Content Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = content_model(movie_list=fav_movies,
                                                            top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


        if sys == 'Collaborative Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = collab_model(movie_list=fav_movies,
                                                           top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


    # -------------------------------------------------------------------

    # ------------- SAFE FOR ALTERING/EXTENSION -------------------
    if page_selection == "Solution Overview":
        st.image('resources/imgs/project_blackbox.jpeg',use_column_width=True)
        st.write("## Solutions Overview")
        st.write("In today’s technology driven world, recommender systems are socially and \
        economically critical to ensure that individuals can make optimised choices \
        surrounding the content they engage with on a daily basis. One application where \
        this is especially true is movie recommendations; where intelligent algorithms can \
        help viewers find great titles from tens of thousands of options.")
        st.write("Recommender Systems are a type of information filtering system as they \
        improve the quality of search results and provides items that are more relevant to \
        the search item or are related to the search history of the user.")
        st.write("Let's view a breakdown of these recommender systems as applied in **Project \
        Blackbox** as providing an accurate and robust solutions to this challenge has immense \
        economic potential, with consumers of the system being personalised recommendations - \
        generating platform affinity for the streaming services which best facilitates their \
        audience's viewing.")
        st.image('resources/imgs/Recommender_systems.png',use_column_width=True)

    if page_selection == "Exploratory Data Analysis":
        st.image('resources/imgs/project_blackbox.jpeg',use_column_width=True)
        st.title("Exploratory Data Analysis")
        st.write('The three major feautures included in our Recommender System Algorithms: \
          \n  - Movie Year Release\n  - Movie Genre\n - Movie Director')
        st.write("### Total Movies Released per Year")
        st.image('resources/imgs/yearly_released_movies_1.png',use_column_width=True)


        st.write("### Distribution of Movie Genres")
        st.image('resources/imgs/movie_genre_distribution.png',use_column_width=True)

        st.write("### Top 10 Most Popular Movie Directors")
        st.image('resources/imgs/popular_movie_directors.png',use_column_width=True)

        st.write("## Insights")

        
    if page_selection == "About Us":
        st.title("Meet the Data Science Team")
        st.image('resources/imgs/project_blackbox.jpeg',use_column_width=True)
        components.html(
        """
        <style media="screen">
          .card {
            --card-gradient: rgba(0, 0, 0, 0.8);
            --card-gradient: #5e9ad9, #e271ad;
            // --card-gradient: tomato, orange;
            --card-blend-mode: overlay;
            // --card-blend-mode: multiply;

            background-color: #fff;
            border-radius: 0.5rem;
            box-shadow: 0.05rem 0.1rem 0.3rem -0.03rem rgba(0, 0, 0, 0.45);
            padding-bottom: 1rem;
            background-image: linear-gradient(
              var(--card-gradient),
              white max(9.5rem, 27vh)
            );
            overflow: hidden;

            img {
              border-radius: 0.5rem 0.5rem 0 0;
              width: 100%;
              object-fit: cover;
              // height: max(10rem, 25vh);
              max-height: max(10rem, 30vh);
              aspect-ratio: 4/3;
              mix-blend-mode: var(--card-blend-mode);
              // filter: grayscale(100);

              ~ * {
                margin-left: 1rem;
                margin-right: 1rem;
              }
            }

            > :last-child {
              margin-bottom: 0;
            }

            &:hover,
            &:focus-within {
              --card-gradient: #24a9d5 max(8.5rem, 20vh);
            }
          }

          /* Additional demo display styles */
          * {
            box-sizing: border-box;
          }

          body {
            display: grid;
            place-content: center;
            justify-items: center;
            min-height: 100vh;
            margin: 0;
            padding: 1rem;
            line-height: 1.5;
            font-family: -apple-system, BlinkMacSystemFont, avenir next, avenir,
              helvetica neue, helvetica, Ubuntu, roboto, noto, segoe ui, arial, sans-serif;
            color: #444;
          }

          .card h3 {
            margin-top: 1rem;
            font-size: 1.25rem;
          }

          .card a {
            color: inherit;
          }

          .card-wrapper {
            list-style: none;
            padding: 0;
            margin: 0;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(30ch, 1fr));
            grid-gap: 1.5rem;
            max-width: 100vw;
            width: 120ch;
            padding-left: 1rem;
            padding-right: 1rem;
          }

        </style>
        <ul class="card-wrapper">
          <li class="card">
            <img src='resources/imgs/Njabulo.jpeg' alt=''>
            <h3><a href="">Njabulo Mkhwanazi</a></h3>
            <p>Business Analyst</p>
          </li>

          <li class="card">
            <img src='resources/imgs/Josh.jpeg' alt=''>
            <h3><a href="">Joshan Dooki</a></h3>
            <p>ML Engineer</p>
          </li>

          <li class="card">
            <img src='resources/imgs/Roger.jpeg' alt=''>
            <h3><a href="">Roger Arendse</a></h3>
            <p>Data Scientist</p>
          </li>

          <li class="card">
            <img src='https://images.unsplash.com/photo-1611916656173-875e4277bea6?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=MXwxNDU4OXwwfDF8cmFuZG9tfHx8fHx8fHw&ixlib=rb-1.2.1&q=80&w=400' alt=''>
            <h3><a href="">William Hlatshwayo</a></h3>
            <p>Developer</p>
          </li>

            <li class="card">
            <img src='resources/imgs/Gabrielle.jpeg' alt=''>
            <h3><a href="">Gabrielle Peria</a></h3>
            <p>Statistician</p>
          </li>

          <li class="card">
            <img src='resources/imgs/Wade.jpeg' alt=''>
            <h3><a href="">Wade Jacobs</a></h3>
            <p>Security Specialist</p>
          </li>


        </ul>
        """,
        height=1000, scrolling=True
        )
        # local_css('resources/pages/html_style.css')

    # You may want to add more sections here for aspects such as an EDA,
    # or to provide your business pitch.


if __name__ == '__main__':
    main()
