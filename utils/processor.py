import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from PyPDF2 import PdfReader

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')

class ResumeProcessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))

    def extract_text_from_pdf(self, pdf_path):
        """Extracts text from a PDF file using PyPDF2."""
        text = ""
        try:
            reader = PdfReader(pdf_path)
            for page in reader.pages:
                text += page.extract_text()
        except Exception as e:
            print(f"Error extracting PDF: {e}")
        return text

    def clean_text(self, text):
        """Cleans the extracted text."""
        # Remove URLs
        text = re.sub(r'http\S+\s*', ' ', text)
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        # Convert to lowercase
        text = text.lower()
        # Tokenize
        tokens = word_tokenize(text)
        # Remove stop words and short words
        cleaned_tokens = [w for w in tokens if w not in self.stop_words and len(w) > 2]
        return " ".join(cleaned_tokens)

if __name__ == "__main__":
    # Test
    processor = ResumeProcessor()
    sample_text = "I am a Python Developer with experience in Machine Learning and SQL. Contact me at http://example.com"
    print(f"Original: {sample_text}")
    print(f"Cleaned: {processor.clean_text(sample_text)}")
