# model_training.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import re
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

MODEL_PATH = 'models/resume_model.pkl'
VECTORIZER_PATH = 'models/tfidf_vectorizer.pkl'

# Clean text function
def clean_text(text):
    text = re.sub(r'http\\S+|www\\S+|https\\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^A-Za-z0-9]+', ' ', text)
    text = ' '.join([word for word in text.lower().split() if word not in stop_words])
    return text

# Load and preprocess data
def load_data():
    data = pd.read_csv('UpdatedResumeDataSet.csv')
    data['cleaned_resume'] = data['Resume'].apply(clean_text)
    return data

# Train model
def train_and_save_model():
    data = load_data()
    X_train, X_test, y_train, y_test = train_test_split(data['cleaned_resume'], data['Category'], test_size=0.2, random_state=42)
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    model = LogisticRegression()
    model.fit(X_train_tfidf, y_train)
    
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print("Model and vectorizer saved.")

if __name__ == "__main__":
    train_and_save_model()
