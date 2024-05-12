import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import joblib

# Load the trained model and vectorizer
model = joblib.load('./trained_models/classifier.pkl')
vectorizer = joblib.load('./trained_models/vectorizer.pkl')

def predict_tweet(tweet):
    # Step 1: Preprocess the tweet
    # Apply any specific cleaning needed
    tweet_cleaned = tweet  # Add your cleaning code here if needed

    # Vectorize the tweet using the loaded vectorizer
    tweet_vectorized = vectorizer.transform([tweet_cleaned])

    # Step 2: Make a prediction
    prediction = model.predict(tweet_vectorized)

    # Step 3: Interpret the prediction
    if prediction[0] == 0:
        return "Normal"
    else:
        return "Harmful"

# Example usage
input_tweet = "I Love You"
print("The tweet is classified as:", predict_tweet(input_tweet))
