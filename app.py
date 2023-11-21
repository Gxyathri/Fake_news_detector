from flask import Flask, render_template, request, jsonify
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import numpy as np
from sklearn.base import TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

from gensim.models import Word2Vec
w2v_model = Word2Vec.load(r"----")


app = Flask(__name__)


def predict_fake_news(news_text):
    
    tokenized_news = word_tokenize(news_text.lower())
    embeddings = [w2v_model.wv[word] for word in tokenized_news if word in w2v_model.wv]
    avg_embedding = np.mean(embeddings, axis=0) if embeddings else np.zeros(w2v_model.vector_size)
    X_input = avg_embedding.reshape(1, -1)
    prediction = model.predict(X_input)[0]
    result = "Fake" if prediction == 1 else "Real"
    return result

def predict_fake_news(news_text):
    try:
        
        tokenized_news = word_tokenize(news_text.lower())
        embeddings = [w2v_model.wv[word] for word in tokenized_news if word in w2v_model.wv]

        
        avg_embedding = np.mean(embeddings, axis=0) if embeddings else np.zeros(w2v_model.vector_size)

        
        X_input = avg_embedding.reshape(1, -1)

        
        prediction = model.predict(X_input)[0]

        
        result = "Fake" if prediction == 1 else "Real"

        return result
    except KeyError:
        
        return "Unknown"


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
