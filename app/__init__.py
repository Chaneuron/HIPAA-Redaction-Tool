# C:\Users\drdon\RAG\Redact\app\__init__.py
from flask import Flask
from pathlib import Path

def create_app():
    app = Flask(__name__,
                template_folder=Path(__file__).parent.parent / 'templates',
                static_folder=Path(__file__).parent.parent / 'static')
    app.config.from_object('config.Config')
    return app