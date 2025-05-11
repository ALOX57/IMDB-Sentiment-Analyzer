import streamlit as st
import joblib

# Load the saved model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Streamlit UI
st.title("Movie Review Sentiment Analyzer")

review = st.text_area("Enter a movie review:")

if st.button("Analyze Sentiment"):
    if review.strip() == "":
        st.warning("Please enter a review.")
    else:
        # Convert review to vector
        review_vector = vectorizer.transform([review])

        # Predict probability
        prob = model.predict_proba(review_vector)[0][1]

        # Classify with thresholds
        if prob > 0.75:
            st.success(f"Very Positive ({prob:.2f})")
        elif prob > 0.5:
            st.success(f"Positive ({prob:.2f})")
        elif prob < 0.25:
            st.error(f"Very Negative ({prob:.2f})")
        else:
            st.warning(f"Negative ({prob:.2f})")