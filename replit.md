# Overview

This is a College Event Feedback Sentiment Analysis application designed to analyze feedback from college events such as fests, hackathons, and workshops. The application provides both web-based interfaces (Streamlit and FastAPI) for analyzing sentiment in feedback text, supporting both single feedback analysis and batch processing. It handles multilingual text including Hinglish and uses advanced machine learning models for accurate sentiment classification with confidence calibration.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture
The application provides two distinct frontend approaches:
- **Streamlit Interface**: Python-based web UI for interactive data analysis with built-in caching and session state management
- **HTML/JavaScript Interface**: Traditional web frontend with Chart.js for visualizations, served through FastAPI static file mounting

## Backend Architecture
### Core Framework
- **FastAPI**: RESTful API backend providing endpoints for sentiment analysis, file uploads, and data processing
- **Streamlit**: Alternative web interface with integrated backend functionality

### Machine Learning Pipeline
- **XLM-RoBERTa Model**: Pre-trained transformer model (`cardiffnlp/twitter-xlm-roberta-base-sentiment`) for multilingual sentiment analysis
- **Temperature Scaling**: Custom PyTorch module for confidence calibration to improve prediction reliability
- **Text Preprocessing**: NLTK-based preprocessing with lazy initialization to avoid reload issues

### Data Processing
- **Pandas/NumPy**: Core data manipulation and analysis
- **CSV Processing**: Support for bulk feedback analysis through file uploads
- **Caching Strategy**: Streamlit resource caching for model loading optimization

## Authentication & Security
No authentication system implemented - designed as a standalone analysis tool with CORS middleware for cross-origin requests.

## API Structure
### FastAPI Endpoints
- Single feedback analysis endpoint
- Batch processing endpoint  
- CSV file upload and processing
- Static file serving for frontend assets
- Chart generation with matplotlib backend

### Data Flow
1. Input validation and preprocessing
2. Sentiment analysis using transformer model
3. Confidence calibration through temperature scaling
4. Result aggregation and visualization
5. Response formatting (JSON/HTML)

# External Dependencies

## Machine Learning Models
- **Hugging Face Transformers**: XLM-RoBERTa pre-trained model for sentiment analysis
- **PyTorch**: Deep learning framework for model inference and temperature scaling

## Natural Language Processing
- **NLTK**: Text preprocessing, tokenization, stopword removal, and lemmatization
- **SentencePiece**: Tokenization support for transformer models

## Data Visualization
- **Matplotlib**: Backend chart generation for sentiment distribution plots
- **Chart.js**: Frontend interactive visualizations
- **Plotly**: Advanced interactive plotting for Streamlit interface

## Web Framework Dependencies
- **FastAPI**: Modern async web framework with automatic API documentation
- **Uvicorn**: ASGI server for FastAPI deployment
- **Streamlit**: Rapid web app development framework for data science applications

## Data Processing Libraries
- **Pandas**: DataFrame operations and CSV processing
- **NumPy**: Numerical computations and array operations
- **scikit-learn**: Machine learning utilities and model evaluation tools
- **SciPy**: Scientific computing functions including softmax for probability calculations

## Additional Utilities
- **python-multipart**: File upload handling in FastAPI
- **aiofiles**: Async file operations
- **python-dotenv**: Environment variable management