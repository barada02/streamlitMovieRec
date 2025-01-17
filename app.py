import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests

# Set page config
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# Sample movie data (in real app, you would use a larger dataset)
movies_data = {
    'title': [
        'The Shawshank Redemption', 'The Godfather', 'The Dark Knight',
        'Pulp Fiction', 'Fight Club', 'Inception', 'The Matrix',
        'Goodfellas', 'The Silence of the Lambs', 'Interstellar'
    ],
    'genre': [
        'Drama', 'Crime, Drama', 'Action, Crime, Drama',
        'Crime, Drama', 'Drama', 'Action, Adventure, Sci-Fi',
        'Action, Sci-Fi', 'Biography, Crime, Drama',
        'Crime, Drama, Thriller', 'Adventure, Drama, Sci-Fi'
    ],
    'rating': [9.3, 9.2, 9.0, 8.9, 8.8, 8.8, 8.7, 8.7, 8.6, 8.6],
    'description': [
        'Two imprisoned men bond over a number of years.',
        'The aging patriarch of an organized crime dynasty transfers control.',
        'The Dark Knight of Gotham City begins his war on crime.',
        'The lives of two mob hitmen, a boxer, and a pair of diner bandits intertwine.',
        'An insomniac office worker and a devil-may-care soapmaker form an underground fight club.',
        'A thief who steals corporate secrets through dream-sharing technology.',
        'A computer programmer discovers a mysterious world.',
        'The story of Henry Hill and his life in the mob.',
        'A young FBI cadet must receive help from an incarcerated cannibal killer.',
        'A team of explorers travel through a wormhole in space.'
    ],
    'year': [1994, 1972, 2008, 1994, 1999, 2010, 1999, 1990, 1991, 2014],
    'poster_path': [
        '/q6y0Go1tsGEsmtFryDOJo3dEmqu.jpg',
        '/3bhkrj58Vtu7enYsRolD1fZdja1.jpg',
        '/qJ2tW6WMUDux911r6m7haRef0WH.jpg',
        '/d5iIlFn5s0ImszYzBPb8JPIfbXD.jpg',
        '/pB8BM7pdSp6B6Ih7QZ4DrQ3PmJK.jpg',
        '/9gk7adHYeDvHkCSEqAvQNLV5Uge.jpg',
        '/f89U3ADr1oiB1s9GkdPOEpXUk5H.jpg',
        '/aKuFiU82s5ISJpGZp7YkIr3kCUd.jpg',
        '/rplLJ2hPcOQmkFhTqUte0MkEaO2.jpg',
        '/gEU2QniE6E77NI6lCU6MxlNBvIx.jpg'
    ]
}

# Convert to DataFrame
movies_df = pd.DataFrame(movies_data)

# TMDB configuration
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"

# Function to display rating as stars
def display_rating_stars(rating):
    full_stars = int(rating // 1)
    half_star = rating % 1 >= 0.5
    empty_stars = 5 - full_stars - (1 if half_star else 0)
    
    stars = "⭐" * full_stars
    if half_star:
        stars += "½"
    return stars

# Custom CSS with dark mode support
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .movie-container {
        background-color: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .movie-title {
        color: #FF4B4B;
        font-size: 1.5rem;
        margin-bottom: 0.5rem;
    }
    .movie-info {
        color: rgba(255, 255, 255, 0.8);
        font-size: 0.9rem;
    }
    .movie-poster {
        border-radius: 10px;
        max-width: 100%;
        height: auto;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and introduction
st.title("🎬 Movie Recommendation System")
st.markdown("### Discover Your Next Favorite Movie!")

# Sidebar
with st.sidebar:
    st.header("Filters")
    selected_genre = st.multiselect(
        "Select Genre(s)",
        sorted(list(set([genre.strip() for genres in movies_df['genre'] for genre in genres.split(',')])))
    )
    
    min_rating = st.slider("Minimum Rating", 0.0, 10.0, 8.0, 0.1)
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("This app helps you discover movies based on your preferences. It uses content-based filtering to recommend similar movies.")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Movie Search")
    search_query = st.text_input("Enter a movie title or keywords")
    
    if search_query:
        filtered_movies = movies_df[
            movies_df['title'].str.contains(search_query, case=False) |
            movies_df['description'].str.contains(search_query, case=False)
        ]
        
        if not filtered_movies.empty:
            for _, movie in filtered_movies.iterrows():
                with st.container():
                    cols = st.columns([1, 2])
                    with cols[0]:
                        st.image(f"{TMDB_IMAGE_BASE_URL}{movie['poster_path']}", use_column_width=True)
                    with cols[1]:
                        st.markdown(f"### {movie['title']} ({movie['year']})")
                        st.markdown(f"**Rating:** {display_rating_stars(movie['rating'])} ({movie['rating']})")
                        st.markdown(f"**Genre:** {movie['genre']}")
                        st.markdown(f"**Description:** {movie['description']}")
                st.markdown("---")
        else:
            st.info("No movies found matching your search.")

with col2:
    st.subheader("Top Rated Movies")
    top_movies = movies_df[movies_df['rating'] >= min_rating]
    
    if selected_genre:
        top_movies = top_movies[top_movies['genre'].apply(
            lambda x: any(genre.strip() in x for genre in selected_genre)
        )]
    
    for _, movie in top_movies.head().iterrows():
        with st.container():
            st.image(f"{TMDB_IMAGE_BASE_URL}{movie['poster_path']}", use_column_width=True)
            st.markdown(f"**{movie['title']}** ({movie['year']})")
            st.markdown(display_rating_stars(movie['rating']))

# Movie recommendation function
def get_recommendations(movie_title):
    vectorizer = CountVectorizer(stop_words='english')
    content_matrix = vectorizer.fit_transform(movies_df['description'])
    cosine_sim = cosine_similarity(content_matrix, content_matrix)
    
    idx = movies_df[movies_df['title'] == movie_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:4]  # Get top 3 similar movies
    movie_indices = [i[0] for i in sim_scores]
    
    return movies_df.iloc[movie_indices]

# Recommendations section
st.markdown("---")
st.subheader("Get Personalized Recommendations")
selected_movie = st.selectbox("Select a movie you like", movies_df['title'])

if st.button("Get Recommendations"):
    recommendations = get_recommendations(selected_movie)
    st.write("Based on your selection, you might also like:")
    
    for _, movie in recommendations.iterrows():
        with st.container():
            cols = st.columns([1, 2])
            with cols[0]:
                st.image(f"{TMDB_IMAGE_BASE_URL}{movie['poster_path']}", use_column_width=True)
            with cols[1]:
                st.markdown(f"### {movie['title']} ({movie['year']})")
                st.markdown(f"**Rating:** {display_rating_stars(movie['rating'])} ({movie['rating']})")
                st.markdown(f"**Genre:** {movie['genre']}")
                st.markdown(f"**Description:** {movie['description']}")
        st.markdown("---")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit")
