import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
import numpy as np

df = pd.read_csv("data/tripadvisor_hotel_reviews.csv")
print(df.shape)
print(df.head())

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean_review"] = df["Review"].apply(clean_text)

X_train, X_test, y_train, y_test = train_test_split(
    df["clean_review"], df["Rating"],
    test_size=0.2, random_state=42, stratify=df["Rating"]
)

vectorizer = TfidfVectorizer(max_features=2000, stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train).toarray()
X_test_vec = vectorizer.transform(X_test).toarray()

print(X_train_vec.shape)

y_train_idx = y_train.values - 1
y_test_idx = y_test.values - 1

def one_hot(y, n_classes):
    encoded = np.zeros((len(y), n_classes))
    encoded[np.arange(len(y)), y] = 1
    return encoded

n_classes = 5
y_train_onehot = one_hot(y_train_idx, n_classes)

n_samples, n_features = X_train_vec.shape
W = np.zeros((n_features, n_classes))
b = np.zeros(n_classes)

print(W[:5])
print(y_train_onehot[:5])
print(W.shape, b.shape)
print(y_train_onehot[0], y_train.values[0])

def softmax(scores):
    exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

learning_rate = 0.5
epochs = 350

class_counts = np.bincount(y_train_idx, minlength=n_classes)
class_weights = len(y_train_idx) / (n_classes * class_counts)
sample_weights = class_weights[y_train_idx]

for epoch in range(epochs):
    scores = X_train_vec @ W + b
    probs = softmax(scores)

    per_sample_loss = -np.sum(y_train_onehot * np.log(probs + 1e-9), axis=1)
    loss = np.mean(per_sample_loss * sample_weights)

    error = probs - y_train_onehot
    weighted_error = error * sample_weights[:, np.newaxis]
    dW = X_train_vec.T @ weighted_error / n_samples
    db = np.mean(weighted_error, axis=0)

    W -= learning_rate * dW
    b -= learning_rate * db

    if epoch % 50 == 0:
        print(f"epoch {epoch}, loss {loss:.4f}")

test_scores = X_test_vec @ W + b
test_probs = softmax(test_scores)
y_pred = np.argmax(test_probs, axis=1) + 1

print(classification_report(y_test, y_pred))
