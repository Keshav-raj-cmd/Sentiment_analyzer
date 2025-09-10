from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, Response
import pandas as pd
import json
import os
from typing import List, Dict, Optional
import uvicorn
from model import sentiment_analyzer
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import io
import base64

# Initialize FastAPI app
app = FastAPI(title="College Event Feedback Sentiment Analysis", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global variable to store analysis results
analysis_results = []

@app.on_event("startup")
async def startup_event():
    """Initialize the sentiment analyzer on startup"""
    print("Loading sentiment analysis model...")
    success = sentiment_analyzer.load_model()
    if not success:
        print("Failed to load model!")
    else:
        print("Model loaded successfully!")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main HTML page"""
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>HTML file not found. Please ensure index.html exists.</h1>")

@app.post("/api/analyze-single")
async def analyze_single_feedback(feedback: str = Form(...)):
    """Analyze sentiment of a single feedback text"""
    try:
        if not feedback.strip():
            raise HTTPException(status_code=400, detail="Feedback text cannot be empty")
        
        result = sentiment_analyzer.analyze_single_feedback(feedback)
        
        # Store result for later use
        global analysis_results
        analysis_results.append(result)
        
        return {
            "success": True,
            "result": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing feedback: {str(e)}")

@app.post("/api/analyze-batch")
async def analyze_batch_feedback(feedbacks: List[str]):
    """Analyze sentiment of multiple feedback texts"""
    try:
        if not feedbacks:
            raise HTTPException(status_code=400, detail="No feedback texts provided")
        
        results = sentiment_analyzer.analyze_batch_feedback(feedbacks)
        
        # Store results for later use
        global analysis_results
        analysis_results.extend(results)
        
        return {
            "success": True,
            "results": results,
            "count": len(results)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing feedbacks: {str(e)}")

@app.post("/api/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    """Upload and analyze CSV file with feedback data"""
    try:
        print(f"=== BACKEND CSV DEBUG ===")
        print(f"File received: {file}")
        print(f"Filename: {getattr(file, 'filename', None)}")
        print(f"Content-Type: {getattr(file, 'content_type', None)}")
        try:
            print(f"Size: {getattr(file, 'size', 'unknown')}")
        except Exception as e:
            print(f"Could not get file size: {e}")
        print(f"=========================")
        if not file or not getattr(file, 'filename', '').endswith('.csv'):
            print("File is missing or not a CSV!")
            raise HTTPException(status_code=400, detail="File must be a CSV")

        # Read CSV file
        contents = await file.read()
        print(f"Raw file contents (first 200 chars): {contents[:200]!r}")
        try:
            df = pd.read_csv(pd.io.common.StringIO(contents.decode('utf-8')))
        except Exception as e:
            print(f"Error reading CSV: {e}")
            raise HTTPException(status_code=400, detail=f"Could not read CSV: {e}")

        print(f"CSV columns: {df.columns.tolist()}")
        # Check if Feedback column exists
        if 'Feedback' not in df.columns:
            print("CSV missing 'Feedback' column!")
            raise HTTPException(status_code=400, detail="CSV must contain a 'Feedback' column")

        # Analyze feedback
        results_df = sentiment_analyzer.analyze_csv_feedback(df)

        # Apply confidence calibration if enough data
        if len(results_df) >= 20:
            results_df = sentiment_analyzer.calibrate_confidence(results_df)

        # Get sentiment distribution
        sentiment_dist = sentiment_analyzer.get_sentiment_distribution(results_df)

        # Get most common words for each sentiment
        positive_words = sentiment_analyzer.get_sentiment_words(results_df, 'Positive', 10)
        negative_words = sentiment_analyzer.get_sentiment_words(results_df, 'Negative', 10)

        # Store results - ensure all values are JSON serializable
        global analysis_results
        analysis_results = results_df.to_dict('records')

        # Convert sentiment distribution to regular Python dict
        sentiment_dist_clean = {str(k): int(v) for k, v in sentiment_dist.items()}

        print("CSV upload and analysis successful.")
        return {
            "success": True,
            "results": results_df.to_dict('records'),
            "sentiment_distribution": sentiment_dist_clean,
            "positive_words": positive_words,
            "negative_words": negative_words,
            "total_count": len(results_df)
        }
    except Exception as e:
        print(f"Error in upload_csv: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing CSV: {str(e)}")

@app.get("/api/sentiment-distribution")
async def get_sentiment_distribution():
    """Get sentiment distribution of all analyzed feedback"""
    try:
        global analysis_results
        if not analysis_results:
            return {"success": True, "distribution": {}}
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(analysis_results)
        distribution = sentiment_analyzer.get_sentiment_distribution(df)
        
        return {
            "success": True,
            "distribution": distribution
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting distribution: {str(e)}")

@app.get("/api/word-analysis")
async def get_word_analysis():
    """Get word frequency analysis for positive and negative sentiments"""
    try:
        global analysis_results
        if not analysis_results:
            return {"success": True, "positive_words": [], "negative_words": []}
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(analysis_results)
        
        positive_words = sentiment_analyzer.get_sentiment_words(df, 'Positive', 10)
        negative_words = sentiment_analyzer.get_sentiment_words(df, 'Negative', 10)
        
        return {
            "success": True,
            "positive_words": positive_words,
            "negative_words": negative_words
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting word analysis: {str(e)}")

@app.get("/api/results")
async def get_all_results():
    """Get all analysis results"""
    try:
        global analysis_results
        return {
            "success": True,
            "results": analysis_results,
            "count": len(analysis_results)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting results: {str(e)}")

@app.delete("/api/clear-results")
async def clear_results():
    """Clear all analysis results"""
    try:
        global analysis_results
        analysis_results = []
        return {"success": True, "message": "Results cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing results: {str(e)}")

@app.get("/api/pie-chart")
async def get_pie_chart():
    """Generate and return pie chart as base64 image"""
    try:
        global analysis_results
        if not analysis_results:
            # Return empty chart if no data
            return {"success": True, "image": None, "message": "No data available"}
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(analysis_results)
        distribution = sentiment_analyzer.get_sentiment_distribution(df)
        
        if not distribution:
            return {"success": True, "image": None, "message": "No sentiment data available"}
        
        # Create pie chart
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Define colors for sentiments
        colors = {
            'Positive': '#48bb78',
            'Neutral': '#ed8936', 
            'Negative': '#f56565'
        }
        
        labels = list(distribution.keys())
        sizes = list(distribution.values())
        chart_colors = [colors.get(label, '#718096') for label in labels]
        
        # Create pie chart
        wedges, texts, autotexts = ax.pie(
            sizes, 
            labels=labels, 
            colors=chart_colors,
            autopct='%1.1f%%',
            startangle=90,
            textprops={'fontsize': 12, 'fontweight': 'bold'}
        )
        
        # Customize the chart
        ax.set_title('Sentiment Distribution', fontsize=16, fontweight='bold', pad=20)
        
        # Make the chart look better
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        # Equal aspect ratio ensures that pie is drawn as a circle
        ax.axis('equal')
        
        # Convert plot to base64 image
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)  # Close the figure to free memory
        
        return {
            "success": True,
            "image": f"data:image/png;base64,{image_base64}",
            "distribution": distribution
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating pie chart: {str(e)}")

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": sentiment_analyzer.model is not None}

if __name__ == "__main__":
    # Create static directory if it doesn't exist
    os.makedirs("static", exist_ok=True)
    import uvicorn
    # Run the application with import string for reload compatibility
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
