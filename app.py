import streamlit as st
from predict import predict_sentiment_en, predict_sentiment_nl

st.title("Sentiment Analysis")

language = st.selectbox("Choose a language", ["English", "Dutch"])
text = st.text_area("Enter a review")

if st.button("Predict"):
    rating = predict_sentiment_en(text) if language == "English" else predict_sentiment_nl(text)
    st.write(f"Predicted rating: {rating} star(s)")



