import pandas as pd
import joblib
import logging
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Function to load a joblib file safely
def safe_load_joblib(path):
    try:
        return joblib.load(path)
    except Exception as e:
        logger.error(f"Failed to load joblib file from {path}", exc_info=True)
        return None

# Load the trained model and vectorizer
logger.info("Loading the trained model and TF-IDF vectorizer...")
clf = safe_load_joblib('./trained_models/classifier.pkl')
vectorizer = safe_load_joblib('./trained_models/vectorizer.pkl')
if not clf or not vectorizer:
    logger.error("Failed to load model or vectorizer. Exiting program.")
    exit(1)
logger.info("Model and vectorizer loaded successfully.")

# Load the test datasets
logger.info("Loading the test datasets...")
test_normal = pd.read_csv('./dataset/testing_dataset/testing-normal.csv')
test_harmful = pd.read_csv('./dataset/testing_dataset/testing-harmful.csv')
logger.info("Test datasets loaded successfully.")

# Combine the datasets
logger.info("Combining test datasets...")
test_data = pd.concat([test_normal, test_harmful], ignore_index=True)
logger.info("Test datasets combined successfully.")

# Extract features and labels
X_test = test_data['tweet']  # Assuming the column with tweets is named 'tweet'
y_test = test_data['ground_truth']

# Convert text data into numerical features using the TF-IDF vectorizer
logger.info("Converting test data into numerical features using the TF-IDF vectorizer...")
X_test_tfidf = vectorizer.transform(X_test)

# Predict the labels of the test dataset
logger.info("Predicting the labels of the test dataset...")
y_pred = clf.predict(X_test_tfidf)

# Evaluate the performance of the model
logger.info("Evaluating the performance of the model...")
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['normal', 'harmful'])
conf_matrix = confusion_matrix(y_test, y_pred)

logger.info(f"Accuracy: {accuracy}")
logger.info(f"Classification Report:\n{report}")

# Print the evaluation results
print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{report}")

# Plot the confusion matrix
logger.info("Plotting the confusion matrix...")
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['normal', 'harmful'], yticklabels=['normal', 'harmful'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

# Extracting precision, recall, f1-score for visualization
logger.info("Extracting precision, recall, and f1-score for visualization...")
report_dict = classification_report(y_test, y_pred, target_names=['normal', 'harmful'], output_dict=True)
categories = list(report_dict.keys())[:-3]  # Exclude 'accuracy', 'macro avg', 'weighted avg'
precision = [report_dict[cat]['precision'] for cat in categories]
recall = [report_dict[cat]['recall'] for cat in categories]
f1_score = [report_dict[cat]['f1-score'] for cat in categories]

x = np.arange(len(categories))  # the label locations
width = 0.2  # the width of the bars

# Plotting precision, recall, f1-score
fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width, precision, width, label='Precision')
rects2 = ax.bar(x, recall, width, label='Recall')
rects3 = ax.bar(x + width, f1_score, width, label='F1-Score')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores')
ax.set_title('Scores by category and metric')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()

# Function to label the bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(round(height, 2)),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

fig.tight_layout()

plt.show()
