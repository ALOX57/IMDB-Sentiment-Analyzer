import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Load the dataset from CSV
df = pd.read_csv("IMDB Dataset.csv")

# Convert 'positive'/'negative' into 1 / 10
df["label"] = df["sentiment"].map({"positive": 1, "negative": 0})