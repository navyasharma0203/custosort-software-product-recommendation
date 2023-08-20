from flask import Flask, render_template, request
import numpy as np
from gensim.models import Word2Vec
from tensorflow.keras.models import load_model
import joblib
import torch

app = Flask(__name__)

# Load your Word2Vec model and recommendation model
word2vec_model = Word2Vec.load('flipkart_software.pkl')
recommendation_model = load_model('flipkart_software.pkl')

# Example recommendation data (replace this with actual recommendations)
recommendations = [
    {"product": "Product 1", "map": 0.85, "ndcg": 0.92},
    {"product": "Product 2", "map": 0.78, "ndcg": 0.85},
    # Add more recommendations here
]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Assuming user input is received from a form field named 'user_input'
        user_input = request.form.get('user_input')

        # Process user input and generate recommendations
        # Replace this with your actual recommendation generation logic
        user_recommendations = recommendations

        return render_template('user.html', user_input=user_input, recommendations=user_recommendations)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
