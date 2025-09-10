#!/usr/bin/env python3
"""
Startup script for College Event Feedback Sentiment Analysis Application
"""

import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    try:
        import fastapi
        import uvicorn
        import pandas
        import numpy
        import transformers
        import torch
        import nltk
        print("✅ All required packages are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        return False

def download_nltk_data():
    """Download required NLTK data"""
    try:
        import nltk
        print("📥 Downloading NLTK data...")
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        print("✅ NLTK data downloaded successfully")
    except Exception as e:
        print(f"⚠️  Warning: Could not download NLTK data: {e}")
        print("The application will try to download it automatically on first run.")

def create_static_directory():
    """Ensure static directory exists"""
    static_dir = Path("static")
    if not static_dir.exists():
        static_dir.mkdir()
        print("📁 Created static directory")

def start_server():
    """Start the FastAPI server"""
    print("🚀 Starting College Event Feedback Sentiment Analysis Application...")
    print("📊 Loading ML model (this may take a few minutes on first run)...")
    print("🌐 Server will be available at: http://localhost:8000")
    print("📱 Open your browser to start analyzing feedback!")
    print("\n" + "="*60)
    print("Press Ctrl+C to stop the server")
    print("="*60 + "\n")
    
    try:
        # Start the server
        subprocess.run([sys.executable, "main.py"], check=True)
    except KeyboardInterrupt:
        print("\n👋 Server stopped. Thank you for using the application!")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        return False
    
    return True

def main():
    """Main function"""
    print("🎓 College Event Feedback Sentiment Analysis")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("main.py").exists():
        print("❌ Error: main.py not found. Please run this script from the project directory.")
        return
    
    # Check requirements
    if not check_requirements():
        return
    
    # Download NLTK data
    download_nltk_data()
    
    # Create static directory
    create_static_directory()
    
    # Wait a moment then open browser
    def open_browser():
        time.sleep(3)  # Wait for server to start
        try:
            webbrowser.open("http://localhost:8000")
        except:
            pass  # Browser opening is optional
    
    # Start server
    start_server()

if __name__ == "__main__":
    main()



