from flask import Flask, request, jsonify, render_template
import joblib
import os
import re
from PyPDF2 import PdfReader
from nltk.corpus import stopwords
import nltk
import numpy as np

app = Flask(__name__)

MODEL_PATH = 'models/resume_model.pkl'
VECTORIZER_PATH = 'models/tfidf_vectorizer.pkl'

# Download stopwords if not already available
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load model and vectorizer with error handling
try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
except Exception as e:
    print("Error loading model or vectorizer:", e)
    model = None
    vectorizer = None

def clean_text(text):
    # Basic text cleaning function
    text = re.sub(r'http\\S+|www\\S+|https\\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^A-Za-z0-9]+', ' ', text)
    text = ' '.join([word for word in text.lower().split() if word not in stop_words])
    return text

# Extract text from PDF using PyPDF2
def extract_text_from_pdf(pdf_file):
    try:
        text = ""
        reader = PdfReader(pdf_file)
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        print("Error extracting text from PDF:", e)
        return None

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if file is in the request
        if 'resume' not in request.files:
            print("No file part in request")
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['resume']
        
        # Check if a valid PDF file was uploaded
        if file.filename == '':
            print("No selected file")
            return jsonify({'error': 'No selected file'}), 400
        
        # Extract text from PDF
        resume_text = extract_text_from_pdf(file)
        
        if resume_text is None:
            print("Failed to extract text from PDF")
            return jsonify({'error': 'Failed to extract text from PDF'}), 500
        
        # Clean and predict
        cleaned_text = clean_text(resume_text)
        resume_tfidf = vectorizer.transform([cleaned_text])
        
        # Get the predicted category and confidence score
        prediction = model.predict(resume_tfidf)
        prediction_prob = model.predict_proba(resume_tfidf)
        
        # Find the confidence score for the predicted category
        confidence_score = np.max(prediction_prob) * 100  # Convert to percentage
        
        return jsonify({
            'category': prediction[0],
            'confidence_score': round(confidence_score, 2)  # Rounded to 2 decimal places
        })
    except Exception as e:
        print("Prediction error:", e)
        return jsonify({'error': 'An error occurred during prediction'}), 500

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
