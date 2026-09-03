import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

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
