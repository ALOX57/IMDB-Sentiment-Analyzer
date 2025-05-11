import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Load the dataset from CSV
df = pd.read_csv("IMDB Dataset.csv")

# Convert 'positive'/'negative' into 1 / 10
df["label"] = df["sentiment"].map({"positive": 1, "negative": 0})

x = df["review"]
y = df["label"]

# Convert text into TF-IDF vectors
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
x_vec = vectorizer.fit_transform(x)

# Train a logistic regression model on the vectorized data
model = LogisticRegression()
model.fit(x_vec, y)