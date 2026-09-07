import joblib
import numpy as np
import re

model_data = joblib.load("model.pkl")
W = model_data["W"]
b = model_data["b"]
vectorizer = model_data["vectorizer"]

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def softmax(scores):
    exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

def predict_sentiment(text):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned]).toarray()
    scores = vec @ W + b
    probs = softmax(scores)
    return np.argmax(probs) + 1

if __name__ == "__main__":
    review = "The room was disgusting and the staff was rude"
    print(predict_sentiment(review), "star(s)")