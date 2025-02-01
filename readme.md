# HIPAA-Compliant Document Redaction Tool

A web application for redacting sensitive information from documents and images with OCR capabilities.

## Features
- PDF and image file processing
- OCR text extraction
- Automated PHI (Protected Health Information) redaction
- Document comparison
- Detailed metrics and analysis

## Setup
1. Install required packages:
```pip install -r requirements.txt```

2. Install Tesseract OCR:
   - Download from: https://github.com/UB-Mannheim/tesseract/wiki
   - Install to: `C:\Program Files\Tesseract-OCR\`

3. Run the application:
```python app.py```

4. Access the application at: http://localhost:5000