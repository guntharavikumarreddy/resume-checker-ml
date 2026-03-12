import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ResumeMatcher:
    def __init__(self):
        self.skills_list = [
            'Python', 'Java', 'SQL', 'Machine Learning', 'React', 'Node.js', 
            'Deep Learning', 'C++', 'JavaScript', 'AWS', 'Docker', 'Kubernetes',
            'TensorFlow', 'PyTorch', 'Data Analysis', 'Flask', 'Django', 'Spring',
            'PostgreSQL', 'MongoDB', 'Angular', 'Vue.js', 'HTML', 'CSS', 'Pandas',
            'Numpy', 'Scikit-learn', 'NLP', 'Computer Vision', 'DevOps'
        ]

    def extract_skills(self, text):
        """Extracts skills from text based on a predefined list."""
        found_skills = []
        for skill in self.skills_list:
            # Use regex to find whole words case-insensitively
            pattern = r'\b' + re.escape(skill) + r'\b'
            if re.search(pattern, text, re.IGNORECASE):
                found_skills.append(skill)
        return list(set(found_skills))

    def calculate_similarity(self, resume_text, job_description):
        """Calculates cosine similarity between resume and job description."""
        try:
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([resume_text, job_description])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
            return round(similarity[0][0] * 100, 2)
        except Exception:
            return 0.0

if __name__ == "__main__":
    matcher = ResumeMatcher()
    resume = "Python Developer with Machine Learning experience"
    jd = "Looking for a Python Developer with Machine Learning and SQL knowledge"
    print(f"Skills: {matcher.extract_skills(resume)}")
    print(f"Similarity: {matcher.calculate_similarity(resume, jd)}%")
