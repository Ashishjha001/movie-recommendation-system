import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import streamlit as st
import requests
import os
import sys

# First, check if pickle files exist
def check_pickle_files():
    print("Checking for pickle files...")
    movies_pkl_exists = os.path.isfile('movies.pkl')
    similarity_pkl_exists = os.path.isfile('similarity.pkl')
    
    print(f"movies.pkl exists: {movies_pkl_exists}")
    print(f"similarity.pkl exists: {similarity_pkl_exists}")
    
    return movies_pkl_exists and similarity_pkl_exists

# Data loading and preprocessing with error handling
def load_and_process_data():
    try:
        print("Attempting to load CSV files...")
        # Check if CSV files exist
        if not os.path.isfile('tmdb_5000_movies.csv'):
            print("ERROR: tmdb_5000_movies.csv not found!")
            st.error("tmdb_5000_movies.csv not found. Please ensure the file is in the same directory.")
            return None
            
        if not os.path.isfile('tmdb_5000_credits.csv'):
            print("ERROR: tmdb_5000_credits.csv not found!")
            st.error("tmdb_5000_credits.csv not found. Please ensure the file is in the same directory.")
            return None
        
        # Load datasets
        movies = pd.read_csv('tmdb_5000_movies.csv')
        credits = pd.read_csv('tmdb_5000_credits.csv')
        
        print(f"Loaded movies dataset with {movies.shape[0]} rows and {movies.shape[1]} columns")
        print(f"Loaded credits dataset with {credits.shape[0]} rows and {credits.shape[1]} columns")
        
        # Check for id column in credits dataset
        if 'id' in credits.columns and 'movie_id' not in credits.columns:
            print("Renaming 'id' column to 'movie_id' in credits dataset")
            credits.rename(columns={'id': 'movie_id'}, inplace=True)
        
        # Merge datasets on movie_id if both have it, otherwise try title
        if 'movie_id' in movies.columns and 'movie_id' in credits.columns:
            print("Merging datasets on movie_id")
            movies = movies.merge(credits, on='movie_id')
        else:
            print("Merging datasets on title")
            movies = movies.merge(credits, on='title')
        
        print(f"Merged dataset has {movies.shape[0]} rows and {movies.shape[1]} columns")
        
        # Select relevant features - handle different column names
        if 'id' in movies.columns and 'movie_id' not in movies.columns:
            movies.rename(columns={'id': 'movie_id'}, inplace=True)
            
        movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
        
        # Drop rows with missing values
        before_dropna = movies.shape[0]
        movies.dropna(inplace=True)
        after_dropna = movies.shape[0]
        print(f"Dropped {before_dropna - after_dropna} rows with missing values")
        
        return movies
    except Exception as e:
        print(f"ERROR in load_and_process_data: {str(e)}")
        st.error(f"Error loading data: {str(e)}")
        return None

# Helper functions to parse JSON-like strings in the dataset
def convert_ast(obj):
    try:
        return ast.literal_eval(obj)
    except (ValueError, SyntaxError):
        return []

def get_list_of_names(obj_list, key='name'):
    if isinstance(obj_list, list) and len(obj_list) > 0:
        return [item[key] for item in obj_list if key in item]
    return []

def get_director(crew_data):
    if isinstance(crew_data, list):
        for crew_member in crew_data:
            if crew_member.get('job') == 'Director':
                return crew_member.get('name', '')
    return ''

# Feature extraction with error handling
def extract_features(movies):
    try:
        print("Beginning feature extraction...")
        
        # Convert string representations to Python objects
        print("Converting JSON strings to Python objects...")
        movies['genres'] = movies['genres'].apply(convert_ast)
        movies['keywords'] = movies['keywords'].apply(convert_ast)
        movies['cast'] = movies['cast'].apply(convert_ast)
        movies['crew'] = movies['crew'].apply(convert_ast)
        
        # Extract names from lists
        print("Extracting names from lists...")
        movies['genres'] = movies['genres'].apply(get_list_of_names)
        movies['keywords'] = movies['keywords'].apply(get_list_of_names)
        movies['cast'] = movies['cast'].apply(lambda x: get_list_of_names(x[:3]) if len(x) > 3 else get_list_of_names(x))
        movies['director'] = movies['crew'].apply(get_director)
        
        # Process overview
        print("Processing overview text...")
        movies['overview'] = movies['overview'].apply(lambda x: x.split() if isinstance(x, str) else [])
        
        # Create tags by combining features
        print("Creating combined tags...")
        movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + [[movies['director'][i]] for i in range(len(movies))]
        
        # Convert lists to space-separated strings
        movies['tags'] = movies['tags'].apply(lambda x: ' '.join([str(item).lower().replace(' ', '_') for item in x if isinstance(item, str)]))
        
        # Keep only necessary columns for final dataframe
        final_movies = movies[['movie_id', 'title', 'tags']]
        print(f"Final processed dataset has {final_movies.shape[0]} rows")
        
        return final_movies
    except Exception as e:
        print(f"ERROR in extract_features: {str(e)}")
        st.error(f"Error extracting features: {str(e)}")
        return None

# Vectorization and similarity calculation with error handling
def vectorize_and_calculate_similarity(movies):
    try:
        print("Starting vectorization...")
        # Create CountVectorizer object
        cv = CountVectorizer(max_features=5000, stop_words='english')
        
        # Create vectors from tags
        print("Creating vectors from tags...")
        vectors = cv.fit_transform(movies['tags']).toarray()
        print(f"Created vectors with shape: {vectors.shape}")
        
        # Calculate cosine similarity
        print("Calculating cosine similarity...")
        similarity = cosine_similarity(vectors)
        print(f"Created similarity matrix with shape: {similarity.shape}")
        
        return similarity
    except Exception as e:
        print(f"ERROR in vectorize_and_calculate_similarity: {str(e)}")
        st.error(f"Error calculating similarities: {str(e)}")
        return None

# Save processed data and model with error handling
def save_data_and_model(movies, similarity):
    try:
        print("Saving movies dataframe as pickle...")
        pickle.dump(movies.to_dict(), open('movies.pkl', 'wb'))
        
        print("Saving similarity matrix as pickle...")
        pickle.dump(similarity, open('similarity.pkl', 'wb'))
        
        print("Successfully saved pickle files.")
        return True
    except Exception as e:
        print(f"ERROR in save_data_and_model: {str(e)}")
        st.error(f"Error saving data: {str(e)}")
        return False

# Movie recommendation function
def recommend_movies(movie_name, movies, similarity):
    try:
        # Find the index of the movie
        idx = movies[movies['title'] == movie_name].index[0]
        
        # Get similarity scores
        sim_scores = list(enumerate(similarity[idx]))
        
        # Sort movies based on similarity
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get top 5 similar movies (excluding the movie itself)
        sim_scores = sim_scores[1:6]
        
        # Get movie indices
        movie_indices = [i[0] for i in sim_scores]
        
        # Return recommended movies
        return movies['title'].iloc[movie_indices].tolist(), [movies['movie_id'].iloc[i] for i in movie_indices]
    except Exception as e:
        print(f"ERROR in recommend_movies: {str(e)}")
        st.error(f"Error generating recommendations: {str(e)}")
        return [], []

# Function to fetch movie poster from TMDB
def fetch_poster(movie_id):
    try:
        # Using provided API key
        API_KEY = "6b454548a561fff10100055f090a068a"
        response = requests.get(f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}&language=en-US")
        data = response.json()
        return "https://image.tmdb.org/t/p/w500/" + data['poster_path']
    except:
        return "https://via.placeholder.com/300x450?text=No+Poster+Available"

# Main function to run the entire pipeline with error handling
def build_recommendation_system():
    print("Starting the recommendation system build process...")
    
    # Check if files already exist
    if check_pickle_files():
        print("Pickle files already exist. If you want to rebuild them, delete the existing files first.")
    
    print("Loading and processing data...")
    movies = load_and_process_data()
    
    if movies is None:
        print("Failed to load and process data.")
        return False
    
    print("Extracting features...")
    final_movies = extract_features(movies)
    
    if final_movies is None:
        print("Failed to extract features.")
        return False
    
    print("Vectorizing and calculating similarity...")
    similarity = vectorize_and_calculate_similarity(final_movies)
    
    if similarity is None:
        print("Failed to calculate similarity.")
        return False
    
    print("Saving data and model...")
    if not save_data_and_model(final_movies, similarity):
        print("Failed to save data and model.")
        return False
    
    print("Done! The recommendation system is ready.")
    return True

# Streamlit web application with error handling
def create_streamlit_app():
    # Page configuration
    st.set_page_config(
        page_title="Movie Recommendation System",
        page_icon="🎬",
        layout="wide"
    )
    
    # Custom CSS for styling
    st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;
        }
        .header {
            padding: 1.5rem 0;
            background: linear-gradient(90deg, rgba(103,58,183,1) 0%, rgba(63,81,181,1) 100%);
            color: white;
            border-radius: 10px;
            margin-bottom: 2rem;
            text-align: center;
        }
        .movie-card {
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s ease;
            height: 100%;
            overflow: hidden;
        }
        .movie-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2);
        }
        .movie-poster {
            width: 100%;
            border-top-left-radius: 10px;
            border-top-right-radius: 10px;
        }
        .movie-info {
            padding: 1rem;
        }
        .movie-title {
            font-weight: bold;
            font-size: 1.2rem;
            margin-bottom: 0.5rem;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .search-container {
            background-color: white;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 2rem;
        }
        .footer {
            margin-top: 3rem;
            text-align: center;
            padding: 1rem;
            color: #6c757d;
        }
        .stButton>button {
            background-color: #673ab7;
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            font-weight: bold;
            border-radius: 5px;
            transition: background-color 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #5e35b1;
        }
        .debug-info {
            margin-top: 2rem;
            padding: 1rem;
            background-color: #f0f0f0;
            border-radius: 5px;
            font-family: monospace;
            font-size: 0.9rem;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<div class="header"><h1>🎬 Movie Recommendation System</h1><p>Find your next favorite movie</p></div>', unsafe_allow_html=True)
    
    # Initialize session state to track app state
    if 'app_state' not in st.session_state:
        st.session_state.app_state = 'loading'
    
    # Check for pickle files
    movies_file_exists = os.path.isfile('movies.pkl')
    similarity_file_exists = os.path.isfile('similarity.pkl')
    
    # If pickle files don't exist, build the system
    if not movies_file_exists or not similarity_file_exists:
        st.warning("Movie data files not found. Building recommendation system...")
        
        with st.spinner("Processing movie data... This may take a few minutes."):
            success = build_recommendation_system()
            
        if not success:
            st.error("Failed to build recommendation system. Check the logs for details.")
            
            # Show debug information
            st.markdown('<div class="debug-info">', unsafe_allow_html=True)
            st.subheader("Debugging Information")
            st.write("Please check if your dataset files are named correctly and are in the correct format.")
            st.write("Expected files:")
            st.write("- tmdb_5000_movies.csv")
            st.write("- tmdb_5000_credits.csv")
            
            # Check for the files
            st.write(f"tmdb_5000_movies.csv exists: {os.path.isfile('tmdb_5000_movies.csv')}")
            st.write(f"tmdb_5000_credits.csv exists: {os.path.isfile('tmdb_5000_credits.csv')}")
            
            # List files in the current directory
            st.write("Files in current directory:")
            st.write(os.listdir('.'))
            st.markdown('</div>', unsafe_allow_html=True)
            
            return
    
    # Load data
    try:
        movies_dict = pickle.load(open('movies.pkl', 'rb'))
        movies = pd.DataFrame(movies_dict)
        similarity = pickle.load(open('similarity.pkl', 'rb'))
        st.session_state.app_state = 'ready'
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        
        # Show debug information
        st.markdown('<div class="debug-info">', unsafe_allow_html=True)
        st.subheader("Debugging Information")
        st.write(f"Error details: {str(e)}")
        st.write(f"movies.pkl exists: {movies_file_exists}")
        st.write(f"similarity.pkl exists: {similarity_file_exists}")
        
        # If files exist but can't be loaded, they might be corrupted
        if movies_file_exists and similarity_file_exists:
            st.write("Pickle files exist but could not be loaded. They might be corrupted.")
            if st.button("Rebuild Recommendation System"):
                # Remove old files
                try:
                    os.remove('movies.pkl')
                    os.remove('similarity.pkl')
                    st.success("Old files removed. Rebuilding system...")
                    success = build_recommendation_system()
                    if success:
                        st.success("Recommendation system rebuilt successfully!")
                        st.rerun()
                    else:
                        st.error("Failed to rebuild recommendation system.")
                except Exception as e:
                    st.error(f"Error removing files: {str(e)}")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    # Search section
    st.markdown('<div class="search-container">', unsafe_allow_html=True)
    st.subheader("What movie do you like?")
    
    selected_movie = st.selectbox(
        "Select a movie you enjoyed watching", 
        movies['title'].values
    )
    
    if st.button('Show Recommendations'):
        with st.spinner('Finding movies for you...'):
            names, ids = recommend_movies(selected_movie, movies, similarity)
            
        st.markdown('</div>', unsafe_allow_html=True)  # Close search container
        
        if not names:
            st.error("Could not generate recommendations. Please try another movie.")
            return
            
        # Display recommendations
        st.subheader("You might also like these movies:")
        
        cols = st.columns(5)
        for i, (name, movie_id) in enumerate(zip(names, ids)):
            poster_url = fetch_poster(movie_id)
            with cols[i]:
                st.markdown(f"""
                <div class="movie-card">
                    <img class="movie-poster" src="{poster_url}" alt="{name}">
                    <div class="movie-info">
                        <div class="movie-title">{name}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.markdown('</div>', unsafe_allow_html=True)  # Close search container
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>Powered by TMDB API | Built with Streamlit and Python</p>
    </div>
    """, unsafe_allow_html=True)

# Main function that you would run
if __name__ == "__main__":
    # Redirect print statements to stderr so they show up in the Streamlit logs
    sys.stdout = sys.stderr
    print("Starting Movie Recommendation System...")
    
    # Create the Streamlit app
    create_streamlit_app()

