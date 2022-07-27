# Streamlit-based Recommender System
#### EXPLORE Data Science Academy Unsupervised Predict

#Project Blackbox
![Project Blackbox Logo](1. Data/project_blackbox.jpeg)
![Movie_Recommendations](resources/imgs/Image_header.png)

In today’s technology driven world, recommender systems are socially and economically critical to ensure that individuals can make optimised choices surrounding the content they engage with on a daily basis. One application where this is especially true is movie recommendations; where intelligent algorithms can help viewers find great titles from tens of thousands of options.

With this context, EDSA is challenging you to construct a recommendation algorithm based on content or collaborative filtering, capable of accurately predicting how a user will rate a movie they have not yet viewed, based on their historical preferences.

Providing an accurate and robust solution to this challenge has immense economic potential, with users of the system being personalised recommendations - generating platform affinity for the streaming services which best facilitates their audience's viewing. See figure below adapted from [Kaggle](https://www.kaggle.com/competitions/edsa-movie-recommendation-2022/overview).
![Recommender Systems](https://static.observableusercontent.com/files/80ae6cf8ccb082c09784430b22545e429c1b31cf547e492da71a91b33d68448c1dbaf0c3a1693eab7bc44a148facfc95e78acd017053b69b80d9d3bc9cbcc38b)

## Usage Instructions

1. Ensure that you have the prerequisite Python libraries installed on your local machine:

 ```bash
 pip install -U streamlit numpy pandas scikit-learn
 conda install -c conda-forge scikit-surprise
 ```

 2. Clone the repo to your local machine.

 ```bash
 git clone https://github.com/RogerA11/TeamCBB5.git
 ```  

 3. Navigate to the base of the cloned repo, and start the Streamlit app.

 ```bash
 cd TeamCBB5/4. Streamlit Application/
 streamlit run edsa_recommender.py
 ```
 
 ## About the repo
 This repo is a combiantion of [Explore](https://www.google.com/url?q=https://github.com/Explore-AI/unsupervised-predict-streamlit-template&sa=D&source=editors&ust=1658949076133651&usg=AOvVaw2jDFruKE89iOkKd38DVc1s) template repo with a [streamlit](https://streamlit.io/) app framework do deliver a recommendation engine to improve personalised user experience and drive economic benefits and the [Jupyter Notebook](https://jupyter.org/) that was instrumental in releasing the final product
