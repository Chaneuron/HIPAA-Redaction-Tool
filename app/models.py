from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import bcrypt
import jwt
import os
from time import time

db = SQLAlchemy()

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    registration_date = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    is_active = db.Column(db.Boolean, default=True)
    
    def set_password(self, password):
        # Hash password with bcrypt
        salt = bcrypt.gensalt()
        self.password_hash = bcrypt.hashpw(password.encode('utf-8'), salt)
    
    def check_password(self, password):
        # Verify password
        return bcrypt.checkpw(password.encode('utf-8'), self.password_hash)
    
    def generate_auth_token(self, expires_in=3600):
        # Generate JWT token
        return jwt.encode(
            {'id': self.id, 'exp': time() + expires_in},
            os.getenv('SECRET_KEY', 'dev-key'),
            algorithm='HS256'
        )
    
    @staticmethod
    def verify_auth_token(token):
        try:
            data = jwt.decode(
                token,
                os.getenv('SECRET_KEY', 'dev-key'),
                algorithms=['HS256']
            )
            return User.query.get(data['id'])
        except:
            return None

class Document(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    processed_filename = db.Column(db.String(255))
    upload_date = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    user = db.relationship('User', backref=db.backref('documents', lazy=True))
    status = db.Column(db.String(20), default='pending')  # pending, completed, failed
    
    def __repr__(self):
        return f'<Document {self.filename}>'

class AuditLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    action = db.Column(db.String(50), nullable=False)
    details = db.Column(db.Text)
    document_id = db.Column(db.Integer, db.ForeignKey('document.id'))
    
    def __repr__(self):
        return f'<AuditLog {self.action} by User {self.user_id}>'

def init_db(app):
    db.init_app(app)
    with app.app_context():
        db.create_all()