import streamlit as st
import joblib

# Load the saved model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Streamlit UI
st.title("Movie Review Sentiment Analyzer")

review = st.text_area("Enter a movie review:")