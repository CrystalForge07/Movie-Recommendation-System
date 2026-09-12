import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# Load movies and tags
movies = pd.read_csv("data/ml-1m/ml-32m/movies.csv")
tags = pd.read_csv("data/ml-1m/ml-32m/tags.csv")

# Preparing tags
tags["tag"] = tags["tag"].fillna("").astype(str)

tags_grouped = (
    tags.groupby("movieId")["tag"]
    .apply(" ".join)
    .reset_index()
)

# Adding tags to movies
movies = movies.merge(
    tags_grouped,
    on="movieId",
    how="left"
)

movies["tag"] = movies["tag"].fillna("")

# Creating movie profiles (Genres + title + tag)
movies["profile"] = (
    movies["genres"].str.replace("|", " ", regex=False) + " "
    + movies["genres"].str.replace("|", " ", regex=False) + " "
    + movies["title"] + " "
    + movies["tag"]
)


embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2",
    device="cuda"
)

# Generating embeddings
movie_vectors = embedding_model.encode(
    movies["profile"].tolist(),
    show_progress_bar=True
)

# Save embeddings
np.save("movie_embeddings.npy", movie_vectors)

print("\nEmbeddings saved successfully!")