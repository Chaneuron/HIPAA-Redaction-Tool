from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash, session
import os
from werkzeug.utils import secure_filename
import pytesseract
from PIL import Image
import fitz  # PyMuPDF for PDF text extraction
import re
import spacy
from thefuzz import fuzz
import time
from difflib import SequenceMatcher
import traceback
from pathlib import Path
from datetime import timedelta

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Add at the top of app.py after your imports
#DEMO_ACCESS = {'granted': False}  # Simple state holder

app = Flask(__name__, template_folder='templates')

@app.after_request
def after_request(response):
    response.headers['ngrok-skip-browser-warning'] = 'any value'
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST'
    return response

app.secret_key = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'pdf'} 



# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Helper function to check file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']



#@app.route('/')
#def index():
 #   if not DEMO_ACCESS['granted']:
 #       return render_template('register.html')
 #   return render_template('index.html')

@app.route('/')
def index():
    return render_template('index.html')

@app.after_request
def after_request(response):
    response.headers.add('ngrok-skip-browser-warning', 'true')
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', '*')
    response.headers.add('Access-Control-Allow-Methods', '*')
    return response

#@app.route('/register')
#def register():
#    return render_template('register.html')

#@app.route('/check_access', methods=['POST'])
#def check_access():
#    code = request.form.get('access_code')
#    if code == "HIPAA2024":
        #DEMO_ACCESS['granted'] = True
#        return render_template('index.html')
#    return render_template('register.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    start_time = time.time()
    
    #if not DEMO_ACCESS['granted']:
    #    return render_template('register.html')
    
    try:
        if 'file' not in request.files:
            flash('No file uploaded. Please try again.', 'error')
            return render_template('index.html')

        file = request.files['file']
        print(f"File name: {file.filename}")

        if file.filename == '':
            print("Empty filename")
            flash('No file selected. Please choose a file.', 'error')
            return render_template('index.html')

        if file and allowed_file(file.filename):
            print("File is allowed, proceeding with processing")

            # Save uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            print(f"File saved to: {filepath}")

            flash('File uploaded. Starting text extraction...', 'info')

            # Get user inputs for redaction
            phi_fields = {
                "name": request.form.get('name', '').strip(),
                "dob": request.form.get('dob', '').strip(),
                "age": request.form.get('age', '').strip(),
                "address": request.form.get('address', '').strip(),
                "email": request.form.get('email', '').strip(),
                "phone": request.form.get('phone', '').strip(),
                "ssn": request.form.get('ssn', '').strip(),
                "medical_record": request.form.get('medical_record', '').strip(),
                "insurance_id": request.form.get('insurance_id', '').strip(),
                "account_number": request.form.get('account_number', '').strip(),
                "certificate_number": request.form.get('certificate_number', '').strip(),
                "vehicle_id": request.form.get('vehicle_id', '').strip(),
                "device_id": request.form.get('device_id', '').strip(),
                "url": request.form.get('url', '').strip(),
                "ip_address": request.form.get('ip_address', '').strip(),
                "biometric_data": request.form.get('biometric_data', '').strip(),
                "photo": request.form.get('photo', '').strip(),
                "other_identifier": request.form.get('other_identifier', '').strip(),
            }
            print(f"PHI fields received: {phi_fields}")

            # Extract text based on file type
            if filename.endswith('.pdf'):
                text = extract_text_from_pdf(filepath)
                print("Extracted text from PDF")
            else:
                text = perform_ocr(filepath)
                print("Performed OCR on image")

            if not text:
                print("No text extracted from file")
                flash('No text could be extracted from the file.', 'error')
                return render_template('index.html')

            print("Text extraction successful, length:", len(text))
            flash('Text extraction complete. Starting redaction...', 'info')

            # Perform redaction
            redacted_text, redacted_words = redact_sensitive_info(text, phi_fields)
            print("Redaction complete")

            flash('Redaction complete. Calculating metrics...', 'info')

            # Calculate metrics
            metrics = calculate_metrics(text, redacted_text, phi_fields)
            print(f"Metrics calculated: {metrics}")

            # Save redacted text to file
            output_filename = f"{os.path.splitext(filename)[0]}_redacted.txt"
            output_filepath = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

            try:
                with open(output_filepath, 'w', encoding='utf-8') as f:
                    f.write(redacted_text)
                print(f"Redacted text saved to: {output_filepath}")
                flash(f'Redacted file saved as {output_filename}', 'success')
            except Exception as e:
                print(f"Error saving redacted file: {str(e)}")
                flash(f'Error saving redacted file: {str(e)}', 'error')

            # Initialize file_stats with first file
            file_stats = {'file1': get_file_stats(redacted_text)}
            
            # Handle comparison file if provided
            compare_file = request.files.get('compare_file')
            if compare_file and compare_file.filename:
                try:
                    print("Processing comparison file")
                    compare_filename = secure_filename(compare_file.filename)
                    compare_filepath = os.path.join(app.config['UPLOAD_FOLDER'], compare_filename)
                    compare_file.save(compare_filepath)
                    print(f"Comparison file saved to: {compare_filepath}")

                    # Read comparison file content with different encodings
                    encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
                    compare_text = None
                    
                    if compare_filename.endswith('.pdf'):
                        compare_text = extract_text_from_pdf(compare_filepath)
                        print("Extracted text from comparison PDF")
                    else:
                        for encoding in encodings_to_try:
                            try:
                                with open(compare_filepath, 'r', encoding=encoding) as f:
                                    compare_text = f.read()
                                print(f"Successfully read file with {encoding} encoding")
                                break
                            except UnicodeDecodeError:
                                print(f"Failed to read with {encoding} encoding, trying next...")
                                continue

                    if compare_text is None:
                        raise ValueError("Unable to read file with any supported encoding")

                    print(f"Comparison text length: {len(compare_text)}")

                    # Calculate comparison stats
                    comparison_results = compare_texts(redacted_text, compare_text)
                    if comparison_results:
                        file_stats['file2'] = get_file_stats(compare_text)
                        file_stats['comparison'] = comparison_results
                        print(f"Final file stats with comparison: {file_stats}")
                    else:
                        print("No comparison results generated")
                        flash('Could not generate comparison statistics', 'warning')

                except Exception as e:
                    print(f"Error processing comparison file: {str(e)}")
                    flash(f'Error processing comparison file: {str(e)}', 'warning')
                    print(f"Comparison error traceback: {traceback.format_exc()}")

            processing_time = time.time() - start_time
            print(f"Total processing time: {processing_time:.2f} seconds")

            return render_template('index.html',
                                metrics=metrics,
                                redacted_text=redacted_text,
                                output_filename=output_filename,
                                file_stats=file_stats,
                                redacted_words=redacted_words,
                                processing_time=processing_time)

        flash('Invalid file type. Please upload a PNG, JPG, JPEG, or PDF file.', 'error')
        return render_template('index.html')

    except Exception as e:
        processing_time = time.time() - start_time
        print(f"Error in upload_file: {str(e)}")
        print(f"Error type: {type(e)}")
        print(f"Processing time until error: {processing_time:.2f} seconds")
        print(f"Traceback: {traceback.format_exc()}")
        flash(f'An error occurred: {str(e)}', 'error')
        return render_template('index.html')



@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename, as_attachment=True)


# OCR function for images
def perform_ocr(image_path):
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        print(f"OCR failed: {e}")
        return ""

def extract_text_from_pdf(pdf_path):
    try:
        text = ""
        pdf_document = fitz.open(pdf_path)
        
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            # Try regular text extraction first
            page_text = page.get_text()
            
            # If no text is extracted or text is very short, try OCR
            if not page_text or len(page_text.strip()) < 50:
                # Convert PDF page to image
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                
                # Perform OCR on the image
                page_text = pytesseract.image_to_string(img)
                print(f"Used OCR for page {page_num + 1} - extracted {len(page_text)} characters")
            
            text += page_text + "\n"
            
        return text
    except Exception as e:
        print(f"PDF extraction failed: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return ""

def get_file_stats(text):
    """Calculate meaningful statistics for a single file"""
    # Split into words, filtering out empty strings and special characters
    words = [word.strip('.,!?()[]{}:;"\'').lower() 
             for word in text.split() 
             if word.strip('.,!?()[]{}:;"\'')]
    
    # Get meaningful lines (non-empty)
    lines = [line for line in text.splitlines() if line.strip()]
    
    # Count redacted instances
    redacted_count = text.count("[REDACTED]")
    
    return {
        'word_count': len(words),
        'unique_words': len(set(words)),
        'line_count': len(lines),
        'redacted_count': redacted_count,
        'average_words_per_line': round(len(words) / len(lines) if lines else 0, 2)
    }

def convert_date_to_patterns(date_str):
    """Convert a date string to multiple format patterns."""
    try:
        from datetime import datetime
        
        # Handle various input formats
        formats = ["%m/%d/%Y", "%m-%d-%Y", "%Y-%m-%d", "%d/%m/%Y", "%B %d, %Y", "%b %d, %Y"]
        parsed_date = None
        
        for fmt in formats:
            try:
                parsed_date = datetime.strptime(date_str, fmt)
                break
            except ValueError:
                continue
                
        if not parsed_date:
            return []

        # Generate different format patterns
        month_num = parsed_date.month
        day_num = parsed_date.day
        year = parsed_date.year
        month_name = parsed_date.strftime("%B")
        month_abbr = parsed_date.strftime("%b")

        # Basic pattern with word boundaries and optional spaces
        patterns = [
            # Exact matches (with word boundaries)
            fr"\b{month_num}/{day_num}/{year}\b",
            fr"\b{month_name}\s+{day_num},?\s+{year}\b",
            fr"\b{month_abbr}\s+{day_num},?\s+{year}\b",
            
            # Case insensitive versions
            fr"(?i)\b{month_name}\s+{day_num},?\s+{year}\b",
            fr"(?i)\b{month_abbr}\s+{day_num},?\s+{year}\b",
            
            # Variations with zero padding
            fr"\b{month_num:02d}/{day_num:02d}/{year}\b",
            
            # Common alternate formats
            fr"\b{year}-{month_num:02d}-{day_num:02d}\b",
            fr"\b{day_num}\s+(?i){month_name}\s+{year}\b",
            fr"\b{day_num}\s+(?i){month_abbr}\s+{year}\b",
            
            # Additional patterns for full month names and abbreviations
            fr"\b{month_name}\s+{day_num:02d},?\s+{year}\b",
            fr"\b{month_abbr}\s+{day_num:02d},?\s+{year}\b",
            fr"\b{day_num:02d}\s+(?i){month_name}\s+{year}\b",
            fr"\b{day_num:02d}\s+(?i){month_abbr}\s+{year}\b",
        ]

        # Add strict number patterns
        strict_number_patterns = [
            # Strict number patterns with exact digit counts
            fr"\b{month_num:02d}/{day_num:02d}/{year:04d}\b",
            fr"\b{month_num:02d}-{day_num:02d}-{year:04d}\b",
            fr"\b{year:04d}-{month_num:02d}-{day_num:02d}\b",
            fr"\b{day_num:02d}/{month_num:02d}/{year:04d}\b",
        ]
        
        # Combine both pattern lists
        patterns.extend(strict_number_patterns)

        return patterns

    except Exception as e:
        return []

def redact_date_from_text(text, date_patterns, redacted_words=None):
    """Apply date redaction patterns to text."""
    redacted_text = text
    original_text = text
    
    for pattern in date_patterns:
        try:
            # Find all matches before replacing
            matches = re.finditer(pattern, redacted_text)
            for match in matches:
                matched_text = match.group()
                if redacted_words is not None:
                    redacted_words.add(matched_text)
            
            redacted_version = re.sub(pattern, "[REDACTED]", redacted_text)
            redacted_text = redacted_version
        except Exception:
            continue
    
    # Additional fuzzy date detection
    words = redacted_text.split()
    for i in range(len(words)):
        for j in range(2, 5):
            if i + j <= len(words):
                phrase = ' '.join(words[i:i+j])
                # Check if phrase looks like it contains a month name
                if any(month.lower() in phrase.lower() for month in [
                    "january", "february", "march", "april", "may", "june",
                    "july", "august", "september", "october", "november", "december"
                ]):
                    if re.search(r'\b\d{4}\b', phrase):
                        if redacted_words is not None:
                            redacted_words.add(phrase)
                        words[i:i+j] = ['[REDACTED]']
                        break
    
    result = ' '.join(words)
    return result

def match_exact_dob(text, dob_str):
    """Only match exact date of birth format MM/DD/YYYY"""
    try:
        from datetime import datetime
        # Parse the input DOB
        dob = datetime.strptime(dob_str, "%m/%d/%Y")
        
        # Create exact pattern for MM/DD/YYYY format
        pattern = fr"\b{dob.month:02d}/{dob.day:02d}/{dob.year:04d}\b"
        
        matches = re.finditer(pattern, text)
        matched_dates = []
        for match in matches:
            matched_dates.append(match.group())
            
        return matched_dates
    except:
        return []

def redact_sensitive_info(text, phi_fields):
    # Split text keeping original line breaks
    paragraphs = text.splitlines(True)  # True keeps the line endings
    redacted_paragraphs = []
    redacted_words = set()
    
    in_header = False
    in_list = False
    prev_line_empty = False
    
    # Check if any PHI fields are populated
    any_phi_fields = any(value.strip() for value in phi_fields.values())
    
    for paragraph in paragraphs:
        original_spacing = ''
        while paragraph.startswith(' '):
            original_spacing += ' '
            paragraph = paragraph[1:]
            
        # Preserve empty lines
        if not paragraph.strip():
            redacted_paragraphs.append(paragraph)
            prev_line_empty = True
            continue
            
        # Identify formatting patterns
        is_date_header = re.match(r'\d{2}/\d{2}/\d{4}\s*-', paragraph)
        is_section_header = paragraph.strip().isupper() and len(paragraph.strip()) > 3
        is_bullet_point = paragraph.strip().startswith('•') or paragraph.strip().startswith('-')
        is_provider_entry = re.match(r'(Provider|Author|Editor):', paragraph)

        if paragraph.strip():  # Only process non-empty paragraphs
            doc = nlp(paragraph)
            redacted_paragraph = paragraph

            # Only process redaction if PHI fields are provided
            if any_phi_fields:
                # Redact user-specified fields
                for field, value in phi_fields.items():
                    if value:
                        values = [v.strip() for v in value.split(';')]
                        for v in values:
                            if v:
                                if field == "name":
                                    name_parts = v.split()
                                    # Create variations of the name
                                    name_variations = set()
                                    
                                    # Original format
                                    name_variations.add(v)
                                    
                                    # First Last and FIRST LAST
                                    if len(name_parts) >= 2:
                                        name_variations.add(f"{name_parts[0]} {name_parts[-1]}")
                                        name_variations.add(f"{name_parts[0].upper()} {name_parts[-1].upper()}")
                                        
                                        # Last, First and LAST, FIRST
                                        name_variations.add(f"{name_parts[-1]}, {name_parts[0]}")
                                        name_variations.add(f"{name_parts[-1].upper()}, {name_parts[0].upper()}")
                                        
                                        # Individual parts and their uppercase versions
                                        for part in name_parts:
                                            name_variations.add(part)
                                            name_variations.add(part.upper())
                                    
                                    # Add all variations to redacted_words
                                    redacted_words.update(name_variations)
                                    
                                    # Create patterns for each variation
                                    for variation in name_variations:
                                        pattern = r'\b' + re.escape(variation) + r'\b'
                                        redacted_paragraph = re.sub(pattern, "[REDACTED]", 
                                                                  redacted_paragraph, 
                                                                  flags=re.IGNORECASE)
                                    
                                    # Apply fuzzy matching with stricter threshold
                                    words = redacted_paragraph.split()
                                    for i in range(len(words)):
                                        for j in range(2, 5):
                                            if i + j <= len(words):
                                                phrase = ' '.join(words[i:i+j])
                                                if fuzz.ratio(v.lower(), phrase.lower()) > 85:
                                                    redacted_words.add(phrase)
                                                    words[i:i+j] = ['[REDACTED]']
                                                    break
                                    
                                    redacted_paragraph = ' '.join(words)
                                
                                elif field == "dob":
                                    date_patterns = convert_date_to_patterns(v)
                                    redacted_words.add(v)
                                    # Only match exact date formats, no partial matches
                                    for pattern in date_patterns:
                                        matches = re.finditer(pattern, redacted_paragraph)
                                        for match in matches:
                                            matched_text = match.group()
                                            if '/' in matched_text or '-' in matched_text:
                                                parts = re.split(r'[/-]', matched_text)
                                                if len(parts) == 3 and all(part.isdigit() for part in parts):
                                                    redacted_words.add(matched_text)
                                                    redacted_paragraph = redacted_paragraph.replace(matched_text, "[REDACTED]")

                                elif field == "other_identifier":
                                    redacted_words.add(v)
                                    
                                    # Try to parse as date first
                                    try:
                                        date_patterns = convert_date_to_patterns(v)
                                        if date_patterns:
                                            redacted_paragraph = redact_date_from_text(redacted_paragraph, date_patterns, redacted_words)
                                    except:
                                        pass  # Not a valid date, continue with other checks
                                    
                                    # Check if it's a number
                                    if v.replace('.', '').replace('-', '').isdigit():
                                        pattern = r'\b' + re.escape(v) + r'\b'
                                        redacted_paragraph = re.sub(pattern, "[REDACTED]", redacted_paragraph)
                                    
                                    # Treat as regular text with fuzzy matching
                                    words = redacted_paragraph.split()
                                    for i in range(len(words)):
                                        # Single word matching
                                        if i < len(words):  # Add this check
                                            if fuzz.ratio(v.lower(), words[i].lower()) > 70:
                                                redacted_words.add(words[i])
                                                words[i] = '[REDACTED]'
                                        
                                        # Multi-word matching
                                        for j in range(2, min(5, len(words) - i + 1)):
                                            if i + j <= len(words):  # Add this check
                                                phrase = ' '.join(words[i:i+j])
                                                if fuzz.ratio(v.lower(), phrase.lower()) > 85:
                                                    redacted_words.add(phrase)
                                                    words[i:i+j] = ['[REDACTED]']
                                                    break
                                    
                                    redacted_paragraph = ' '.join(words)
                                    
                                    # Final exact match pass
                                    pattern = re.compile(r'\b' + re.escape(v) + r'\b', re.IGNORECASE)
                                    redacted_paragraph = pattern.sub("[REDACTED]", redacted_paragraph)
                                
                                else:
                                    redacted_words.add(v)
                                    pattern = re.compile(r'\b' + re.escape(v) + r'\b', re.IGNORECASE)
                                    redacted_paragraph = pattern.sub("[REDACTED]", redacted_paragraph)

                # Handle dates from NER if date fields were provided
                if phi_fields.get('dob') or phi_fields.get('date'):
                    for ent in doc.ents:
                        if ent.label_ == "DATE":
                            redacted_words.add(ent.text)
                            redacted_paragraph = redacted_paragraph.replace(ent.text, "[REDACTED]")

            # Preserve formatting based on line type
            if is_date_header:
                redacted_paragraph = original_spacing + redacted_paragraph
                if not prev_line_empty:
                    redacted_paragraph = '\n' + redacted_paragraph
            
            elif is_section_header:
                redacted_paragraph = original_spacing + redacted_paragraph.upper()
                if not prev_line_empty:
                    redacted_paragraph = '\n\n' + redacted_paragraph
            
            elif is_bullet_point:
                redacted_paragraph = original_spacing + '• ' + redacted_paragraph.lstrip('•').lstrip('-').lstrip()
            
            elif is_provider_entry:
                redacted_paragraph = original_spacing + redacted_paragraph
                if not prev_line_empty:
                    redacted_paragraph = '\n' + redacted_paragraph

            redacted_paragraphs.append(redacted_paragraph)
            prev_line_empty = False

    # Join with proper spacing
    final_text = ''
    for i, para in enumerate(redacted_paragraphs):
        if i > 0:
            # Add extra newline before headers and after sections
            if (any(header in para for header in ['DISCHARGE', 'ADMISSION', 'DIAGNOSES']) or
                para.strip().isupper()):
                final_text += '\n'
        final_text += para

    return final_text, sorted(list(redacted_words))

def calculate_metrics(original_text, redacted_text, phi_fields):
    metrics = {}
    
    # Count redacted instances
    redacted_count = redacted_text.count("[REDACTED]")
    if redacted_count > 0:
        metrics["total_redactions"] = redacted_count
    
    # Calculate modification percentage
    total_words = len(original_text.split())
    if total_words > 0:
        modified_percentage = (redacted_count / total_words) * 100
        metrics["text_modified_percentage"] = round(modified_percentage, 2)
    
    # Check which fields were redacted
    redacted_fields = []
    for field, value in phi_fields.items():
        if value and value.strip() and value.strip().lower() not in redacted_text.lower():
            redacted_fields.append(field)
    
    if redacted_fields:
        metrics["fields_redacted"] = len(redacted_fields)
        metrics["redacted_field_types"] = ", ".join(redacted_fields)
    
    return metrics

def compare_texts(text1, text2):
    """
    Compare two texts and return various similarity metrics.
    """
    # Print debug information
    print("Starting text comparison...")
    print(f"Text 1 length: {len(text1)}")
    print(f"Text 2 length: {len(text2)}")

    try:
        # Get word sets for comparison (handle empty or None inputs)
        if not text1 or not text2:
            print("One or both texts are empty")
            return None

        # Split into words and clean them
        words1 = set(word.strip('.,!?()[]{}:;"\'').lower() 
                    for word in text1.split() 
                    if word.strip('.,!?()[]{}:;"\''))
        words2 = set(word.strip('.,!?()[]{}:;"\'').lower() 
                    for word in text2.split() 
                    if word.strip('.,!?()[]{}:;"\''))
        
        print(f"Words in text 1: {len(words1)}")
        print(f"Words in text 2: {len(words2)}")

        # Calculate intersection and unique words
        common_words = words1.intersection(words2)
        unique_to_file1 = len(words1 - words2)
        unique_to_file2 = len(words2 - words1)
        
        # Calculate word-based similarity
        total_unique_words = len(words1.union(words2))
        word_similarity = (len(common_words) / total_unique_words * 100) if total_unique_words > 0 else 0
        
        # Calculate overall content similarity using SequenceMatcher
        import difflib
        content_similarity = difflib.SequenceMatcher(None, text1, text2).ratio() * 100
        
        results = {
            'content_similarity': round(content_similarity, 2),
            'word_similarity': round(word_similarity, 2),
            'common_words': len(common_words),
            'unique_to_file1': unique_to_file1,
            'unique_to_file2': unique_to_file2
        }
        
        print("Comparison results:", results)
        return results

    except Exception as e:
        print(f"Error in compare_texts: {str(e)}")
        return None





if __name__ == '__main__':
    app.run(debug=True, port=5000)