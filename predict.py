import joblib
import numpy as np
import re

model_en = joblib.load("model.pkl")
W_en, b_en, vectorizer_en = model_en["W"], model_en["b"], model_en["vectorizer"]

model_nl = joblib.load("model_nl.pkl")
W_nl, b_nl, vectorizer_nl = model_nl["W_nl"], model_nl["b_nl"], model_nl["vectorizer_nl"]

def clean_text_en(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def clean_text_nl(text):
    text = text.lower()
    text = re.sub(r"[^a-zà-ÿ\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def softmax(scores):
    exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

def predict_sentiment_en(text):
    cleaned = clean_text_en(text)
    vec = vectorizer_en.transform([cleaned]).toarray()
    scores = vec @ W_en + b_en
    return np.argmax(softmax(scores)) + 1

def predict_sentiment_nl(text):
    cleaned = clean_text_nl(text)
    vec = vectorizer_nl.transform([cleaned]).toarray()
    scores = vec @ W_nl + b_nl
    return np.argmax(softmax(scores)) + 1

if __name__ == "__main__":
    print(predict_sentiment_en("The room was disgusting and the staff was rude"))
    print(predict_sentiment_nl("Het eten was koud en de bediening was traag"))