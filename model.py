import numpy as np
from gensim.models import Word2Vec
from tensorflow.keras.models import load_model
import joblib
import torch

model_path = 'flipkart_software.pkl'

# Load Word2Vec model
def load_word2vec_model(model_path):
    word2vec_model = Word2Vec.load(model_path)
    return word2vec_model

# Load recommendation model
def load_recommendation_model(model_path):
    recommendation_model = load_model(model_path)
    return recommendation_model

# Generate recommendations using the loaded models
def generate_recommendations(user_input, word2vec_model, recommendation_model):
    # Assuming user_input is a query from the user
    # Preprocess user input and generate Word2Vec embeddings
    user_input_embeddings = word2vec_model.wv[user_input.split()]

    # Make recommendations using the recommendation model
    # Replace this with your actual recommendation generation logic
    recommendations = [
        {"product": "Product 1", "map": 0.85, "ndcg": 0.92},
        {"product": "Product 2", "map": 0.78, "ndcg": 0.85},
        {"product": "Product 3", "map": 0.71, "ndcg": 0.65},
        {"product": "Product 4", "map": 0.62, "ndcg": 0.45},
        {"product": "Product 5", "map": 0.54, "ndcg": 0.43},
        # Add more recommendations here
    ]

    return recommendations

