# Sentiment Analysis Flask App

This Flask application allows users to predict the sentiment of a given text review using a pre-trained sentiment analysis model. Users can also report incorrect predictions to improve the model over time.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Training](#model-training)
- [Feedback Integration](#feedback-integration)


## Features

- Predict sentiment of text reviews (positive/negative).
- Report incorrect predictions to improve the model.
- Monitor and retrain the model with new feedback.

## Installation

1. **Clone the repository:**

    ```sh
    git clone https://github.com/WWWW-W/Sentiment_Prediction.git
    cd sentiment_analysis_flask
    ```

2. **Create and activate a virtual environment:**

    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required packages:**

    ```sh
    pip install -r requirements.txt
    ```

4. **Download and place the pre-trained model:**

    Download the pre-trained model `best_sentiment_analysis_model.h5` and place it in the `model/` directory.

5. **Run the Flask app:**s

    ```sh
    python app.py
    ```

6. **Open the app in your browser:**

    Go to `http://127.0.0.1:5000` to use the app.

## Usage

1. **Predict Sentiment:**
    - Enter a text review in the input field.
    - Click on the "Predict" button to get the sentiment prediction.

2. **Report Incorrect Prediction:**
    - If the prediction is incorrect, click on the "Report Incorrect Prediction" button.
    - The review will be saved into the excel file for future model retraining.

