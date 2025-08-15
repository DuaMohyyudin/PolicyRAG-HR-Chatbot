import os

class Config:
    # Google Sheets Configuration
    GOOGLE_SHEETS_ID = ""
    SHEET_NAME = "Sheet1"
    CREDENTIALS_PATH = "credentials.json"
    
    # Policy Document
    POLICY_PDF_PATH = os.path.join(os.path.dirname(__file__), "hr_policies.pdf")
    # Remove the asterisks (**)
    # Use CSV format (most reliable)
    EXCEL_URL = "https://example.com/path/to/excel.csv"
    
    # Authentication
    MAX_LOGIN_ATTEMPTS = 3
    
    # RAG Configuration
    CHUNK_SIZE = 200
    CHUNK_OVERLAP = 100
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    SEARCH_K =3
    
    # Gemini Configuration
    GEMINI_MODEL = "gemini-2.5-flash"
    GEMINI_TEMPERATURE = 0.3
    GEMINI_API_KEY = ""  # Replace with your actual API key
    
    # Leave Application
    MIN_LEAVE_DAYS = 0.5
    MAX_LEAVE_DAYS = 30
    
    HR_EMAIL = "hr@example.com"
    