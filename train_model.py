import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import os

# Ensure models directory exists
models_dir = os.path.join(os.getcwd(), 'models')
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# Synthetic Data for Demonstration
data = {
    'resume_text': [
        "python data science machine learning sql pandas numpy",
        "react node js javascript html css frontend web developer",
        "java spring boot hibernate sql microservices backend",
        "aws cloud devops docker kubernetes jenkins terraform",
        "hr management recruitment payroll employee relations",
        "sales marketing lead generation customer relationship",
        "python django flask postgresql sql api",
        "machine learning deep learning pytorch tensorflow computer vision",
        "react native mobile app ios android",
        "sql oracle database administrator dba performance tuning",
        "project management agile scrum jira budgeting",
        "technical writing documentation user manuals api guides",
        "ui ux design figma adobe xd wireframing prototyping",
        "cybersecurity network security penetration testing firewall",
        "data analyst tableau power bi excel sql dashboard"
    ],
    'category': [
        'Data Science', 'Frontend Developer', 'Backend Developer', 'DevOps Engineeer', 
        'HR', 'Sales/Marketing', 'Backend Developer', 'Machine Learning Engineer',
        'Mobile App Developer', 'Database Administrator', 'Project Manager',
        'Technical Writer', 'UI/UX Designer', 'Cybersecurity', 'Data Analyst'
    ]
}

df = pd.DataFrame(data)

def train_and_save_model():
    print("Training models...")
    
    # Feature extraction and model pipeline
    X = df['resume_text']
    y = df['category']
    
    # Vectorizer
    vectorizer = TfidfVectorizer()
    
    # Train Logistic Regression
    lr_model = Pipeline([
        ('vectorizer', vectorizer),
        ('clf', LogisticRegression())
    ])
    lr_model.fit(X, y)
    
    # Train Random Forest
    rf_model = Pipeline([
        ('vectorizer', TfidfVectorizer()),
        ('clf', RandomForestClassifier(n_estimators=100))
    ])
    rf_model.fit(X, y)
    
    # Save models
    with open('models/lr_model.pkl', 'wb') as f:
        pickle.dump(lr_model, f)
    
    with open('models/rf_model.pkl', 'wb') as f:
        pickle.dump(rf_model, f)
        
    print("Models saved to 'models/' directory.")

if __name__ == "__main__":
    train_and_save_model()
