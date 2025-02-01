# C:\Users\drdon\RAG\Redact\app\registration.py
from datetime import datetime, timedelta
from pathlib import Path
import json
import os

class SimpleRegistration:
    def __init__(self):
        # Update to use application root directory
        self.root_dir = Path(__file__).parent.parent
        self.data_dir = self.root_dir / 'data'
        self.logs_dir = self.root_dir / 'logs'
        self.codes_file = self.data_dir / 'access_codes.json'
        
        # Create necessary directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Create codes file if it doesn't exist
        if not self.codes_file.exists():
            with open(self.codes_file, 'w') as f:
                json.dump({}, f)

    def check_access_code(self, code):
        """Validate access code and check expiration."""
        if not code:
            return False
            
        try:
            # Load valid codes
            with open(self.codes_file, 'r') as f:
                valid_codes = json.load(f)
            
            if code not in valid_codes:
                return False
                
            # Check expiration
            issued_date = datetime.fromisoformat(valid_codes[code]['issued_date'])
            if datetime.now() - issued_date > timedelta(days=30):
                return False
                
            return True
            
        except Exception as e:
            print(f"Error checking access code: {str(e)}")
            return False

    def save_access_attempt(self, code):
        """Log access attempt for auditing."""
        try:
            log_file = self.logs_dir / f'access_log_{datetime.now().strftime("%Y%m")}.txt'
            with open(log_file, 'a') as f:
                f.write(f"{datetime.now().isoformat()}: Access attempt with code {code}\n")
        except Exception as e:
            print(f"Error logging access attempt: {str(e)}")