import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv("data/ml-1m/ml-32m/movies.csv")

tags = pd.read_csv("data/ml-1m/ml-32m/tags.csv")

tags["tag"] = tags["tag"].fillna("").astype(str)

tags_grouped = (
    tags.groupby("movieId")["tag"]
    .apply(" ".join)
    .reset_index()
)

movies = movies.merge(
    tags_grouped,
    on = "movieId",
    how = "left"
)

movies["tag"] = movies["tag"].fillna("")

movies["profile"] = (
    movies["title"] + " "
    + movies["genres"].str.replace("|", " ")
    + " "
    + movies["tag"]
)

print(movies[["title", "genres", "tag", "profile"]].head())

vectorizer = TfidfVectorizer(stop_words="english")

movie_vectors = vectorizer.fit_transform(movies["profile"])

print(movie_vectors.shape)

query = "dark science fiction movie with mystery and a complex story"
query_vector = vectorizer.transform([query])

similarities = cosine_similarity(
    query_vector,
    movie_vectors
).flatten()

top_indices = similarities.argsort()[-10:][::-1]

recommendations = movies.iloc[top_indices][["title","genres"]].copy()

recommendations["similarity"] = similarities[top_indices]

print(recommendations)