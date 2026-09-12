# Movie Recommendation System 
A movie recommendation system that understands natural language user requirements and recommends movies using NLP and similarity based ranking.

## Features -
Natural language movie requests.
Both Traditional and LLM-based approaches.
OpenAI API for understanding user preferences.
Cosine similarity for matching preferences with movies.

## Working -

### Traditional -
User query
TF-IDF representation
Cosine similarity 
Top movie recommendations

### LLM approach -
User query
OpenAI preference extraction
TF-IDF representation
Cosine similarity 
Rating + popularity
Top movie recommendations

## Tech stack -
Python
Pandas
numpy
Scikit-learn
OpenAI API

## Dataset -
This project uses the MovieLens dataset from GroupLens.
The dataset is not included in this repository because of its large size.

## Setup -
Install the required libraries:
```bash
pip install pandas scikit-learn openai python-dotenv numpy
