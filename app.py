import os
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load the saved model
model = tf.keras.models.load_model("model/best_sentiment_analysis_model.h5")

# Text preprocessing and TF-IDF vectorizer setup
nltk.download("stopwords")
nltk.download("punkt")
stop_words = set(stopwords.words("english"))


def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    return " ".join(words)


# Prepare the TF-IDF vectorizer with the same settings used during training
tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))

# Fit the TF-IDF vectorizer on the entire dataset (or a large enough sample)
# Load the dataset
df = pd.read_csv("cleaned_augmented_IMDB_Dataset.csv", encoding="ISO-8859-1")
df["review"] = df["review"].apply(clean_text)
tfidf.fit(df["review"])


def predict_sentiment(review):
    cleaned_review = clean_text(review)
    features = tfidf.transform([cleaned_review]).toarray()
    prediction = model.predict(features)[0][0]
    sentiment = "positive" if prediction > 0.5 else "negative"
    return sentiment


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    review = data.get("review", "")
    sentiment = predict_sentiment(review)
    print(f"predicted sentiment: {sentiment}")
    return jsonify({"sentiment": sentiment, "review": review})


@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.get_json()
    review = data.get("review", "")
    correct_sentiment = data.get("sentiment", "")

    # Debugging: Print the correct sentiment
    print(f"Corrected sentiment: {correct_sentiment}")

    # Append the new feedback to the cleaned_augmented_IMDB_Dataset.csv
    feedback_file = "cleaned_augmented_IMDB_Dataset.csv"
    if not os.path.exists(feedback_file):
        feedback_df = pd.DataFrame(columns=["review", "sentiment"])
    else:
        feedback_df = pd.read_csv(feedback_file)

    new_entry = pd.DataFrame({"review": [review], "sentiment": [correct_sentiment]})
    feedback_df = pd.concat([feedback_df, new_entry], ignore_index=True)
    feedback_df.to_csv(feedback_file, index=False)

    return jsonify(
        {
            "status": "feedback received",
            "review": review,
            "correct_sentiment": correct_sentiment,
        }
    )


# Optionally, add a route to retrain the model periodically or based on a trigger

if __name__ == "__main__":
    app.run(debug=True)

os.environ["WHYLABS_DEFAULT_ORG_ID"] = 'org-U6MJzM'
os.environ["WHYLABS_API_KEY"] = 'zcJPs2g1Rk.Hif8SYpIUXxigCUougwLCvjxMCLdV4l5bMJBxiEyKF8tByQ1gF9l3:org-U6MJzM'
os.environ["WHYLABS_DEFAULT_DATASET_ID"] = 'model-2'

