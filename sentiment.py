import pandas as pd
import re

df = pd.read_csv("data/tripadvisor_hotel_reviews.csv")
print(df.shape)
print(df.head())

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean_review"] = df["Review"].apply(clean_text)
