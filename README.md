# IMDB Sentiment Analyzer 

This project is a lightweight sentiment analysis app that classifies IMDB movie reviews as **positive** or **negative** using a logistic regression model and a TF-IDF vectorizer. The interface is built with **Streamlit** for easy interaction.

---

## Features
- Trained on 50,000 real IMDB reviews
- Clean UI with Streamlit
- Text input → Instant prediction
- Structured for easy deployment (e.g. Azure App Service)

---

## Requirements
- Python 3.10+
- pandas, scikit-learn, joblib, streamlit

Install with:

```bash
pip install -r requirements.txt
```
---
## Dataset

This project uses the [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) from Kaggle.

Please download the CSV manually and place it in your project root folder as:
IMDB Dataset.csv

This file is excluded from version control for size and licensing reasons.
