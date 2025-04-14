import streamlit as st
import pickle
import pandas as pd

# Load data
movies = pickle.load(open('movies.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))

# App setup
st.set_page_config(page_title="Movie Recommender", layout="centered")
st.title("🎬 Movie Recommender System")
st.markdown("Get movie recommendations based on what you like!")

# Dropdown menu to select a movie
movie_list = movies['title'].values
selected_movie = st.selectbox("Select a movie", movie_list)

# Recommend function
def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_movies = [movies.iloc[i[0]].title for i in distances[1:6]]
    return recommended_movies

# Button
if st.button("Show Recommendations"):
    recommendations = recommend(selected_movie)
    st.subheader("You might also like:")
    for i, movie in enumerate(recommendations, 1):
        st.write(f"{i}. {movie}")

