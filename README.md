# Movie Recommendation System 
A movie recommendation system that understands natural language user requirements and recommends movies using NLP and similarity based ranking.

## Features 
- Natural language movie requests.
- Groq API for understanding user preferences.
- Sentence Transformer for generating semantic embeddings.
- Cosine similarity for matching preferences with movies.
- Top movie recommendation based on similarity, rating and popularity.

## Working -

### Traditional -
- User query  
- TF-IDF representation  
- Cosine similarity  
- Top movie recommendations  

### LLM approach -
- User query  
- Groq preference extraction  
- Sentence Transformer embeddings  
- Cosine similarity  
- Rating + popularity  
- Top movie recommendations  

## Tech stack -
- Python  
- Pandas  
- numpy  
- Scikit-learn  
- Sentence Transformers  
- Groq API  

## Dataset -
This project uses the MovieLens dataset from GroupLens.  
The dataset is not included in this repository because of its large size.  

## Setup -
Install the required libraries:
```bash
pip install pandas scikit-learn openai python-dotenv numpy sentence-transformers
```

Generate movie embeddings:
```bash
python generate_embeddings.py
```
