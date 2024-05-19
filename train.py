import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import joblib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Step 1: Load and Label Data
logging.info("Loading training data...")
try:
    normal_train = pd.read_csv('./dataset/training_dataset/normal.csv')
    harmful_train = pd.read_csv('./dataset/training_dataset/harmful.csv')
    normal_train['label'] = 0
    harmful_train['label'] = 1
    logging.info("Data loaded successfully.")
except Exception as e:
    logging.error("Failed to load data: %s", e)
    raise

# Combine the datasets
logging.info("Combining datasets...")
train_df = pd.concat([normal_train, harmful_train], ignore_index=True)
logging.info("Datasets combined successfully.")

# Step 2: Vectorization
logging.info("Vectorizing data...")
vectorizer = TfidfVectorizer(max_features=1000)  # Adjust this value as needed
X_train = vectorizer.fit_transform(train_df['tweet'])
y_train = train_df['label']
logging.info("Data vectorization complete.")

# Step 3: Train the SVM classifier
logging.info("Training the SVM classifier...")
model = SVC(kernel='linear', class_weight='balanced')
try:
    model.fit(X_train, y_train)
    logging.info("Model training complete.")
except Exception as e:
    logging.error("Model training failed: %s", e)
    raise

# Final Step: save your trained model and vectorizer
logging.info("Saving the trained model and vectorizer...")
try:
    joblib.dump(model, './trained_models/classifier.pkl')
    joblib.dump(vectorizer, './trained_models/vectorizer.pkl')
    logging.info("Model and vectorizer saved successfully.")
except Exception as e:
    logging.error("Failed to save model/vectorizer: %s", e)
    raise

print("Model training and saving complete.")