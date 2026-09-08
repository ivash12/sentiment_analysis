import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
import numpy as np
import joblib

df = pd.read_parquet("data/RestoReviewRawdata.parquet")
print(df.shape)
print(df.columns.tolist())
print(df[["reviewText", "reviewScoreOverall"]].head())
print(df["reviewScoreOverall"].unique())

df = df.dropna(subset=["reviewScoreOverall", "reviewText"])
df["Rating"] = (df["reviewScoreOverall"] / 2).round().clip(1, 5).astype(int)

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zà-ÿ\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean_review"] = df["reviewText"].apply(clean_text)
