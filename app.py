from flask import Flask, render_template, request, jsonify
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from transformers import WordEmbeddingsTransformer
import joblib
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

app = Flask(__name__)

w2v_model = Word2Vec.load("-----")
model = joblib.load("-----")

def predict_fake_news(news_text):
    
    tokenized_news = word_tokenize(news_text.lower())
    prediction = model.predict(X_input)[0]
    result = "Fake" if prediction == 1 else "Real"
    return result

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    news_text = request.json['news']
    prediction = predict_fake_news(news_text)

    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
