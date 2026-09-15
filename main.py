import os
import pandas as pd
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

load_dotenv()

# Groq API
client = OpenAI(api_key=os.getenv("GROQ_API_KEY"),base_url="https://api.groq.com/openai/v1")

# Load movies 
movies = pd.read_csv("data/ml-1m/ml-32m/movies.csv")

ratings = pd.read_csv(
    "data/ml-1m/ml-32m/ratings.csv",
    usecols=["movieId", "rating"]
)

# Rating statistics
rating_stats = (
    ratings.groupby("movieId")
    .agg(
        avg_rating=("rating", "mean"),
        rating_count=("rating", "count")
    )
    .reset_index()
)

movies = movies.merge(
    rating_stats,
    on="movieId",
    how="left"
)

# Movies with no ratings
movies["avg_rating"] = movies["avg_rating"].fillna(0)
movies["rating_count"] = movies["rating_count"].fillna(0)

# Reduce effect of very large rating counts
movies["log_rating_count"] = np.log1p(movies["rating_count"])

# Normalize rating and popularity
scaler = MinMaxScaler()

movies[["rating_score", "popularity_score"]] = scaler.fit_transform(
    movies[["avg_rating", "log_rating_count"]]
)

# Load pre-generated movie embeddings (from generate_embeddings.py)
movie_vectors = np.load("movie_embeddings.npy")

# Load Sentence transformer 
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Get user's request
query = input("What kind of movie are you looking for ? \n")

# Extract movie preferences using Groq
response = client.responses.create(
    model="openai/gpt-oss-120b",
    instructions="""You are a movie recommendation preference analyzer.

Convert the user's request into concise movie-related keywords and phrases.

Focus on:
- movie genres
- themes
- tone
- audience
- story characteristics

Avoid unrelated words that could cause false matches.

Return only the keywords and phrases, nothing else.""",
    input=query
)

preferences = response.output_text

print("\nExtracted preferences:")
print(preferences)

# Convert user preferences into an embedding
query_vector = embedding_model.encode([preferences])

# Calculate semantic similarity
similarities = cosine_similarity(query_vector,movie_vectors).flatten()

# Final recommendation score
movies["final_score"] = (
    0.70 * similarities
    + 0.20 * movies["rating_score"]
    + 0.10 * movies["popularity_score"]
)

# Filtering movies with low ratings
movies.loc[movies["rating_count"] < 100, "final_score"] = -1

# Get top 10 recommendations
top_indices = movies["final_score"].argsort()[-10:][::-1]

recommendations = movies.iloc[top_indices][
    [
        "title",
        "genres",
        "avg_rating",
        "rating_count",
        "final_score"
    ]
].copy()

print("\nRecommendations:")
print(recommendations)