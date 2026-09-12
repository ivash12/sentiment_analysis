Sentiment Analysis

A sentiment analysis project that rates any given text from 1 to 5 stars, built from scratch in Python and NumPy. Supports both English and Dutch text, with a Streamlit interface for trying it out live.

Features
- Predicts a 1-5 star sentiment rating for any text, in English or Dutch
- Multiclass softmax regression classifier implemented from scratch with NumPy — no scikit-learn model used for the actual classifier
- TF-IDF text vectorization with language-specific stopword removal
- Class-weighted loss to correct for imbalanced training data
- Simple Streamlit interface for live predictions in either language

Built With
- Python 3.14
- NumPy (model, training, inference)
- scikit-learn (TF-IDF vectorization, train/test split, evaluation metrics)
- pandas
- Streamlit
- NLTK (Dutch stopwords)

How It Works

Text is lowercased, stripped of punctuation, and converted into a TF-IDF vector. A softmax regression model — trained entirely from scratch using gradient descent, cross-entropy loss, and manually implemented backpropagation — converts that vector into a probability for each of the 5 star ratings. The highest-probability class is returned as the predicted rating.

Two separate models are trained: one on English hotel reviews, one on Dutch restaurant reviews. Each has its own text cleaning rules (the Dutch version preserves accented characters like ë and ï) and its own stopword list.

Datasets
- English: TripAdvisor Hotel Reviews (Kaggle)
- Dutch: Dutch Restaurant Reviews (Kaggle)

Installation

1. Clone the repository
```
git clone https://github.com/ivash12/sentiment_analysis.git
cd sentiment_analysis
```

2. Create and activate a virtual environment
```
python -m venv .venv
.venv\Scripts\activate   (Windows)
source .venv/bin/activate   (Mac/Linux)
```

3. Install dependencies
```
pip install -r requirements.txt
```

4. Run the app
```
streamlit run app.py
```
Pretrained models (`model.pkl`, `model_nl.pkl`) are included in the repo, so no training is required to use the app.

5. Open your browser at http://localhost:8501

Retraining the models (optional)

If you want to retrain either model yourself (e.g. on updated data):
```
python sentiment.py
python sentiment_nl.py
```

Known Limitations
- Trained on a limited number of reviews (~50k Dutch, ~20k English) — a baseline, not production-grade accuracy
- Middle ratings (2-3 stars) are harder for the model to distinguish than clear extremes (1 or 5 stars)
- The Dutch model is trained on restaurant review vocabulary and may not generalize well to other domains
- Built as a learning project to understand how classification models work under the hood — not intended for production use

Author

https://github.com/ivash12