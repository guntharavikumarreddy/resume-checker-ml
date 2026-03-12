# Resume Screening System - Documentation

This project is a production-ready **AI-powered Resume Screening System** that uses Machine Learning and Natural Language Processing (NLP) to automate the recruitment process.

## 🏗️ Architecture Overview

The system follows a modular architecture to ensure scalability and maintainability:

1.  **Frontend**: A modern, glassmorphism-style UI built with **HTML/Vanilla CSS** and **Plotly.js** for interactive visualizations.
2.  **Backend**: A **Flask** server that handles file uploads, text extraction, and model inference.
3.  **NLP Pipeline**: Located in `utils/processor.py`, it performs text extraction (using **PyPDF2**), cleaning, tokenization, and stopword removal.
4.  **Skill & Matching Engine**: Located in `utils/matcher.py`, it extracts technical skills using regex patterns and calculates **Cosine Similarity** between resumes and job descriptions using **TF-IDF**.
5.  **ML Classifier**: A **Random Forest** model (trained in `train_model.py`) that predicts the job category/role based on the resume content.

## 🚀 Features

- **PDF Text Extraction**: Reliably extracts content from resumes.
- **Advanced NLP**: Specialized cleaning techniques to remove URLs, special characters, and noise.
- **Skill Extraction**: Automatically identifies technical skills like Python, Java, AWS, etc.
- **Smart JD Matching**: Provides a percentage match score compared to any job description.
- **Predictive Analytics**: Predicts the most likely job category for the candidate.
- **Interactive Dashboard**: Visualizes candidate strengths and match metrics.

## 📂 Project Structure

```text
Resume Checker/
├── app.py              # Main Flask application
├── train_model.py      # Script to train and save ML models
├── requirements.txt    # Dependency list
├── models/             # Contains saved .pkl model files
├── templates/          # HTML templates (index, results)
├── static/             
│   ├── css/            # Custom styling (Glassmorphism)
│   └── uploads/        # Temporary storage for uploaded resumes
└── utils/              
    ├── processor.py    # NLP cleaning and extraction logic
    └── matcher.py      # Skill matching and similarity logic
```

## 🛠️ Setup and Installation

1.  **Clone the Repository**:
    ```bash
    git clone <your-repo-link>
    cd "Resume Checker"
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Train the Model**:
    Execute the training script once to generate the machine learning models.
    ```bash
    python train_model.py
    ```

4.  **Run the Application**:
    ```bash
    python app.py
    ```
    The app will be available at `http://127.0.0.1:5000`.

## 🌐 Deployment Guide

### Deployment on Streamlit (Alternative UI)
If you prefer Streamlit for even faster deployment, you can port the logic in `app.py` to a `streamlit_app.py` file and deploy via:
1.  Pushing code to **GitHub**.
2.  Connecting the repo to **Streamlit Cloud**.

### Deployment via Flask on Heroku/Render
1.  Add a `Procfile`:
    ```text
    web: gunicorn app:app
    ```
2.  Connect your GitHub repository to **Render** or **Heroku**.
3.  Set the environment variable (e.g., `PYTHON_VERSION`).

## 👨‍💻 Machine Learning Details
The classifier uses a **TF-IDF Vectorizer** to transform text into numerical features, followed by a **Random Forest Classifier**. This ensemble method is robust for text classification and provides high accuracy for categorizing resumes into roles like *Data Science*, *Web Development*, etc.
