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

def match_skills(resume_text, job_description_text):
    common_skills = {
        "Python", "Java", "SQL", "Project Management", "Machine Learning",
        "Data Analysis", "Communication", "Leadership", "Problem Solving",
        "Time Management", "JavaScript", "HTML", "CSS", "Git", "Teamwork", "Tableau", "R", "Data mining", "Data Visualization", "NLP",
        "Web developer", "frontend developer", "C++", "C", "Backend developer", "devops", "Ruby", "React", "Mern", "Statistical Analysis", "CNN", "LLM",
        "Ms Excel", "data science", "Problem Solving abilities", "Data Analytics", "MongoDB", "Express.js", "Node.js",
        "Ajax", "MVC", "JQuery", "XML", "WCF", ".NET", "ASP.NET", "Mern Stack"
    }
    resume_skills = extract_skills_from_text(resume_text, common_skills)
    job_description_skills = extract_skills_from_text(job_description_text, common_skills)
    matched_skills = resume_skills.intersection(job_description_skills)
    return list(matched_skills)

def extract_missing_skills(resume_text, job_description_text):
    common_skills = {
        "Python", "Java", "SQL", "Project Management", "Machine Learning",
        "Data Analysis", "Communication", "Leadership", "Problem Solving",
        "Time Management", "JavaScript", "HTML", "CSS", "Git", "Teamwork", "Tableau", "R", "Data mining", "Data Visualization", "NLP",
        "Web developer", "frontend developer", "C++", "C", "Backend developer", "devops", "Ruby", "React", "Mern", "Statistical Analysis", "CNN", "LLM",
        "Ms Excel", "data science", "Problem Solving abilities", "Data Analytics", "MongoDB", "Express.js", "Node.js",
        "Ajax", "MVC", "JQuery", "XML", "WCF", ".NET", "ASP.NET", "Mern Stack"
    }
    resume_skills = extract_skills_from_text(resume_text, common_skills)
    job_description_skills = extract_skills_from_text(job_description_text, common_skills)
    missing_skills = job_description_skills - resume_skills
    return list(missing_skills)

def extract_skills_from_text(text, skill_set):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    found_skills = {skill for skill in skill_set if skill.lower() in text}
    return found_skills

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'resume' not in request.files:
            print("No file part in request")
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['resume']
        job_description = request.form.get('jobDescription', '')

        if file.filename == '':
            print("No selected file")
            return jsonify({'error': 'No selected file'}), 400
        
        resume_text = extract_text_from_pdf(file)
        
        if resume_text is None:
            print("Failed to extract text from PDF")
            return jsonify({'error': 'Failed to extract text from PDF'}), 500
        
        cleaned_resume_text = clean_text(resume_text)
        cleaned_job_description = clean_text(job_description)

        resume_tfidf = vectorizer.transform([cleaned_resume_text])
        prediction = model.predict(resume_tfidf)
        prediction_prob = model.predict_proba(resume_tfidf)
        
        resume_score = np.max(prediction_prob) * 100

        matched_skills = match_skills(cleaned_resume_text, cleaned_job_description)
        missing_skills = extract_missing_skills(cleaned_resume_text, cleaned_job_description)
        
        return jsonify({
            'category': prediction[0],
            'resume_score': round(resume_score, 2),
            'matched_skills': matched_skills,
            'missing_skills': missing_skills
        })
    except Exception as e:
        print("Prediction error:", e)
        return jsonify({'error': 'An error occurred during prediction'}), 500

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
