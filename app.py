import os
import pickle
import pandas as pd
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from werkzeug.utils import secure_filename
from utils.processor import ResumeProcessor
from utils.matcher import ResumeMatcher
import plotly.graph_objects as go
import plotly.express as px
import json

app = Flask(__name__)
app.secret_key = "secret_key_resume_checker"
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# Initialize components
processor = ResumeProcessor()
matcher = ResumeMatcher()

# Load models
def load_models():
    try:
        with open('models/rf_model.pkl', 'rb') as f:
            rf_model = pickle.load(f)
        return rf_model
    except FileNotFoundError:
        return None

model = load_models()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'resume' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['resume']
    jd_text = request.form.get('job_description', '')

    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    if file and file.filename.endswith('.pdf'):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # 1. Extract and Clean Text
        raw_text = processor.extract_text_from_pdf(file_path)
        cleaned_text = processor.clean_text(raw_text)

        # 2. Extract Skills
        skills = matcher.extract_skills(raw_text)

        # 3. Predict Job Category
        prediction = "Model not trained"
        if model:
            prediction = model.predict([cleaned_text])[0]

        # 4. Calculate Similarity Score
        match_score = 0
        status = "Not Selected"
        if jd_text:
            match_score = matcher.calculate_similarity(cleaned_text, jd_text)
            if match_score >= 50:
                status = "Selected"
        else:
            status = "N/A (Provide JD)"

        # 5. Visualizations
        skill_counts = {skill: 1 for skill in skills} # Simple count for now
        fig_skills = px.bar(
            x=list(skill_counts.keys()), 
            y=list(skill_counts.values()),
            labels={'x': 'Skill', 'y': 'Presence'},
            title="Extracted Skills Frequency"
        )
        # graph_json = json.dumps(fig_skills, cls=plotly.utils.PlotlyJSONEncoder) if 'plotly' in globals() else None
        
        return render_template('results.html', 
                               prediction=prediction, 
                               skills=skills, 
                               match_score=match_score,
                               status=status,
                               filename=filename)
    
    flash('Only PDF files are allowed')
    return redirect(url_for('index'))

if __name__ == '__main__':
    # Ensure upload directory exists
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    
    app.run(debug=True)
