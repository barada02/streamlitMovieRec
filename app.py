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

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTitle {
        color: #FF4B4B;
        font-size: 3rem !important;
    }
    .stButton>button {
        background-color: #FF4B4B;
        color: white;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

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
    ]
}

# Convert to DataFrame
movies_df = pd.DataFrame(movies_data)

# Title and introduction
st.title("🎬 Movie Recommendation System")
st.markdown("### Discover Your Next Favorite Movie!")

# Sidebar
st.sidebar.header("Filters")
selected_genre = st.sidebar.multiselect(
    "Select Genre(s)",
    sorted(list(set([genre.strip() for genres in movies_df['genre'] for genre in genres.split(',')])))
)

min_rating = st.sidebar.slider("Minimum Rating", 0.0, 10.0, 8.0, 0.1)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Movie Search")
    search_query = st.text_input("Enter a movie title or keywords")
    
    if search_query:
        # Simple search implementation
        filtered_movies = movies_df[
            movies_df['title'].str.contains(search_query, case=False) |
            movies_df['description'].str.contains(search_query, case=False)
        ]
        
        if not filtered_movies.empty:
            for _, movie in filtered_movies.iterrows():
                with st.expander(f"{movie['title']} ({movie['rating']})"):
                    st.write(f"**Genre:** {movie['genre']}")
                    st.write(f"**Description:** {movie['description']}")
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
        st.write(f"**{movie['title']}** ({movie['rating']})")

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
        with st.expander(f"{movie['title']} ({movie['rating']})"):
            st.write(f"**Genre:** {movie['genre']}")
            st.write(f"**Description:** {movie['description']}")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit")
