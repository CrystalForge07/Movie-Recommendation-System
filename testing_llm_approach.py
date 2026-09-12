import pandas as pd
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
load_dotenv()

client = OpenAI()

# Load movies and tags
movies = pd.read_csv("data/ml-1m/ml-32m/movies.csv")
tags = pd.read_csv("data/ml-1m/ml-32m/tags.csv")

ratings = pd.read_csv(
    "data/ml-1m/ml-32m/ratings.csv",
    usecols=["movieId","rating"]
)

rating_stats = (
    ratings.groupby("movieId")
    .agg(
        avg_rating = ("rating","mean"),
        rating_count = ("rating","count")
    ).reset_index()
)

movies = movies.merge(rating_stats,on="movieId",how="left")

# Movies with no ratings

movies["avg_rating"] = movies["avg_rating"].fillna(0)
movies["rating_count"] = movies["rating_count"].fillna(0)

# Using log fucntion to reduce the effect of very large rating counts.

movies["log_rating_count"] = np.log1p(movies["rating_count"])

# Normalising/scaling the two ratings between 0 and 1

scaler = MinMaxScaler()

movies[["rating_score", "popularity_score"]] = scaler.fit_transform(
    movies[["avg_rating", "log_rating_count"]]
)

# Tags

tags["tag"] = tags["tag"].fillna("").astype(str)

tags_grouped = (
    tags.groupby("movieId")["tag"]
    .apply(" ".join)
    .reset_index()
)

# Adding tags to movie data and filling na with zero

movies = movies.merge(
    tags_grouped,
    on="movieId",
    how="left"
)

movies["tag"] = movies["tag"].fillna("")


# Creating movie profile (title + genres + tags)
# Used genres twice to give it more importance (trick)

movies["profile"] = (
    movies["genres"].str.replace("|", " ", regex=False) + " "
    + movies["genres"].str.replace("|", " ", regex=False) + " "
    + movies["title"] + " "
    + movies["tag"]
)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2",device="cuda")

movie_vectors = np.load("movie_embeddings.npy")  # Data loaded from generate_embeddings.py

query = input("What kind of movie are you looking for ? \n")

response = client.responses.create(
    model = "gpt-5.6-luna",
    instructions="""You are a movie recommendation preference analyzer. 
     Convert the user's request into concise movie-related keywords and phrases.
     Focus on movie genres, themes, tone, audience, and story characteristics.
     Avoid unrelated words that could cause false matches.
     Return only the keywords and phrases, nothing else.""",
    input = query
)

preferences = response.output_text

print("\nExtracted preferences : ")
print(preferences)

query_vector = embedding_model.encode([preferences])

similarities = cosine_similarity(
    query_vector,
    movie_vectors
).flatten()

movies["final_score"] = (
    0.70 * similarities
    + 0.20 * movies["rating_score"]
    + 0.10 * movies["popularity_score"]
)

top_indices = movies["final_score"].argsort()[-10:][::-1]

recommendations = movies.iloc[top_indices][
    ["title", "genres", "avg_rating", "rating_count", "final_score"]
].copy()

print("\nRecommendations:")
print(recommendations)