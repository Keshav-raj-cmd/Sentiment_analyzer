import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict
import re
from collections import Counter

# Page configuration
st.set_page_config(
    page_title="College Event Feedback Analyzer",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

class SimpleSentimentAnalyzer:
    """Simple rule-based sentiment analyzer"""
    
    def __init__(self):
        # Simple word lists for sentiment analysis
        self.positive_words = {
            'amazing', 'excellent', 'great', 'awesome', 'fantastic', 'wonderful', 
            'brilliant', 'outstanding', 'perfect', 'good', 'best', 'love', 
            'enjoy', 'helpful', 'organized', 'interactive', 'knowledgeable',
            'memorable', 'fun', 'interesting'
        }
        
        self.negative_words = {
            'bad', 'terrible', 'awful', 'boring', 'poor', 'disappointed', 
            'waste', 'worst', 'hate', 'useless', 'delayed', 'disorganized',
            'lacking', 'mediocre', 'average', 'could be better', 'improve'
        }
    
    def preprocess_text(self, text: str) -> str:
        """Simple text preprocessing"""
        if not isinstance(text, str):
            return ""
        # Convert to lowercase and remove extra whitespace
        return re.sub(r'\s+', ' ', text.lower().strip())
    
    def analyze_single_feedback(self, text: str) -> Dict:
        """Analyze sentiment of a single feedback text"""
        processed_text = self.preprocess_text(text)
        words = set(processed_text.split())
        
        positive_score = len(words.intersection(self.positive_words))
        negative_score = len(words.intersection(self.negative_words))
        
        # Determine sentiment
        if positive_score > negative_score:
            sentiment = 'Positive'
            confidence = min(0.9, 0.6 + (positive_score - negative_score) * 0.1)
        elif negative_score > positive_score:
            sentiment = 'Negative'
            confidence = min(0.9, 0.6 + (negative_score - positive_score) * 0.1)
        else:
            sentiment = 'Neutral'
            confidence = 0.5
        
        # Create scores dict
        scores = {
            'Positive': confidence if sentiment == 'Positive' else (1 - confidence) / 2,
            'Negative': confidence if sentiment == 'Negative' else (1 - confidence) / 2,
            'Neutral': confidence if sentiment == 'Neutral' else 1 - confidence
        }
        
        return {
            'text': text,
            'sentiment': sentiment,
            'confidence': confidence,
            'scores': scores
        }
    
    def analyze_batch_feedback(self, texts: List[str]) -> List[Dict]:
        """Analyze multiple feedback texts"""
        return [self.analyze_single_feedback(text) for text in texts]
    
    def analyze_csv_feedback(self, df: pd.DataFrame, text_column: str = 'Feedback') -> pd.DataFrame:
        """Analyze feedback from a CSV DataFrame"""
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame")
        
        results = []
        for idx, row in df.iterrows():
            text = str(row[text_column])
            result = self.analyze_single_feedback(text)
            results.append(result)
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Add individual score columns
        scores_df = pd.json_normalize(results_df['scores'])
        results_df = pd.concat([results_df.drop('scores', axis=1), scores_df], axis=1)
        
        return results_df
    
    def get_most_common_words(self, texts: List[str], num_words: int = 10) -> List[tuple]:
        """Get most common words from texts"""
        all_words = []
        stopwords = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an', 'is', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'it', 'he', 'she', 'they', 'we', 'you', 'i', 'me', 'my', 'your', 'his', 'her', 'their', 'our', 'this', 'that', 'these', 'those'}
        
        for text in texts:
            if isinstance(text, str):
                words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
                all_words.extend([word for word in words if word not in stopwords])
        
        return Counter(all_words).most_common(num_words)
    
    def get_sentiment_words(self, results_df: pd.DataFrame, sentiment: str, num_words: int = 10) -> List[tuple]:
        """Get most common words for a specific sentiment"""
        sentiment_texts = results_df[results_df['sentiment'] == sentiment]['text'].tolist()
        return self.get_most_common_words(sentiment_texts, num_words)

# Initialize session state
def init_session_state():
    """Initialize session state variables"""
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = []
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = SimpleSentimentAnalyzer()

def clear_results():
    """Clear all analysis results"""
    st.session_state.analysis_results = []
    st.success("Results cleared successfully!")
    st.rerun()

def analyze_single_feedback(text: str, analyzer: SimpleSentimentAnalyzer) -> Dict:
    """Analyze a single feedback text"""
    try:
        result = analyzer.analyze_single_feedback(text)
        return result
    except Exception as e:
        st.error(f"Error analyzing feedback: {str(e)}")
        return None

def analyze_batch_feedback(texts: List[str], analyzer: SimpleSentimentAnalyzer) -> List[Dict]:
    """Analyze multiple feedback texts"""
    try:
        results = analyzer.analyze_batch_feedback(texts)
        return results
    except Exception as e:
        st.error(f"Error analyzing feedbacks: {str(e)}")
        return []

def create_sentiment_chart(results: List[Dict]):
    """Create sentiment distribution visualization"""
    if not results:
        st.info("No data available for visualization")
        return
    
    # Count sentiments
    sentiment_counts = {}
    for result in results:
        sentiment = result['sentiment']
        sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
    
    if not sentiment_counts:
        st.info("No sentiment data available")
        return
    
    # Create simple bar chart using Streamlit
    st.subheader("Sentiment Distribution")
    chart_data = pd.DataFrame(
        list(sentiment_counts.items()),
        columns=['Sentiment', 'Count']
    )
    st.bar_chart(chart_data.set_index('Sentiment'))
    
    # Display statistics
    total = sum(sentiment_counts.values())
    col1, col2, col3 = st.columns(3)
    
    with col1:
        positive_count = sentiment_counts.get('Positive', 0)
        positive_pct = (positive_count / total * 100) if total > 0 else 0
        st.metric("Positive", f"{positive_count}", f"{positive_pct:.1f}%")
    
    with col2:
        neutral_count = sentiment_counts.get('Neutral', 0)
        neutral_pct = (neutral_count / total * 100) if total > 0 else 0
        st.metric("Neutral", f"{neutral_count}", f"{neutral_pct:.1f}%")
    
    with col3:
        negative_count = sentiment_counts.get('Negative', 0)
        negative_pct = (negative_count / total * 100) if total > 0 else 0
        st.metric("Negative", f"{negative_count}", f"{negative_pct:.1f}%")

def display_word_analysis(results: List[Dict], analyzer: SimpleSentimentAnalyzer):
    """Display word frequency analysis"""
    if not results:
        st.info("No data available for word analysis")
        return
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Get word analysis
    try:
        positive_words = analyzer.get_sentiment_words(df, 'Positive', 10)
        negative_words = analyzer.get_sentiment_words(df, 'Negative', 10)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🟢 Most Common Positive Words")
            if positive_words:
                pos_df = pd.DataFrame(positive_words, columns=['Word', 'Count'])
                st.bar_chart(pos_df.set_index('Word')['Count'])
            else:
                st.info("No positive words found")
        
        with col2:
            st.subheader("🔴 Most Common Negative Words")
            if negative_words:
                neg_df = pd.DataFrame(negative_words, columns=['Word', 'Count'])
                st.bar_chart(neg_df.set_index('Word')['Count'])
            else:
                st.info("No negative words found")
                
    except Exception as e:
        st.error(f"Error in word analysis: {str(e)}")

def display_individual_results(results: List[Dict]):
    """Display individual analysis results"""
    if not results:
        st.info("No individual results to display")
        return
    
    st.subheader("📋 Individual Analysis Results")
    
    # Create DataFrame for display
    display_data = []
    for i, result in enumerate(results, 1):
        # Handle both dict and DataFrame row formats
        if isinstance(result, dict):
            text = result.get('text', 'N/A')
            sentiment = result.get('sentiment', 'N/A')
            confidence = result.get('confidence', 0)
            scores = result.get('scores', {})
        else:
            # Handle DataFrame row format
            text = getattr(result, 'text', 'N/A')
            sentiment = getattr(result, 'sentiment', 'N/A')
            confidence = getattr(result, 'confidence', 0)
            scores = {}
            # Try to get individual score columns
            if hasattr(result, 'Positive'):
                scores['Positive'] = getattr(result, 'Positive', 0)
            if hasattr(result, 'Neutral'):
                scores['Neutral'] = getattr(result, 'Neutral', 0)
            if hasattr(result, 'Negative'):
                scores['Negative'] = getattr(result, 'Negative', 0)
        
        display_data.append({
            '#': i,
            'Feedback': text[:100] + "..." if len(str(text)) > 100 else str(text),
            'Sentiment': sentiment,
            'Confidence': f"{confidence:.3f}" if isinstance(confidence, (int, float)) else str(confidence),
            'Positive Score': f"{scores.get('Positive', 0):.3f}",
            'Neutral Score': f"{scores.get('Neutral', 0):.3f}",
            'Negative Score': f"{scores.get('Negative', 0):.3f}"
        })
    
    df = pd.DataFrame(display_data)
    
    # Style the dataframe
    def color_sentiment(val):
        if val == 'Positive':
            return 'background-color: #d4edda; color: #155724'
        elif val == 'Negative':
            return 'background-color: #f8d7da; color: #721c24'
        elif val == 'Neutral':
            return 'background-color: #fff3cd; color: #856404'
        return ''
    
    styled_df = df.style.applymap(color_sentiment, subset=['Sentiment'])
    st.dataframe(styled_df, use_container_width=True, height=400)

def main():
    """Main application function"""
    init_session_state()
    
    # Header
    st.title("🎓 College Event Feedback Analyzer")
    st.markdown("Analyze sentiment of feedback from college events like fests, hackathons, and workshops")
    
    # Add note about simple analyzer
    st.info("📌 This is a simplified version using rule-based sentiment analysis. For more accurate results, the full ML model version is recommended.")
    
    # Sidebar
    st.sidebar.title("Navigation")
    
    # Analysis method selection
    analysis_method = st.sidebar.selectbox(
        "Choose Analysis Method",
        ["Single Feedback", "Batch Analysis", "CSV Upload"]
    )
    
    # Clear results button
    if st.sidebar.button("🗑️ Clear All Results", type="secondary"):
        clear_results()
    
    # Results summary in sidebar
    if st.session_state.analysis_results:
        st.sidebar.markdown("---")
        st.sidebar.subheader("📊 Current Results")
        total_results = len(st.session_state.analysis_results)
        st.sidebar.metric("Total Analyzed", total_results)
        
        # Quick sentiment breakdown
        sentiment_counts = {}
        for result in st.session_state.analysis_results:
            sentiment = result['sentiment']
            sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
        
        for sentiment, count in sentiment_counts.items():
            st.sidebar.metric(sentiment, count)
    
    # Main content area
    if analysis_method == "Single Feedback":
        st.header("💬 Single Feedback Analysis")
        
        # Input form
        feedback_text = st.text_area(
            "Enter your feedback:",
            placeholder="Enter feedback about the college event (e.g., 'The hackathon was amazing! Great organization and helpful mentors.')",
            height=120
        )
        
        if st.button("🔍 Analyze Sentiment", type="primary"):
            if feedback_text.strip():
                with st.spinner("Analyzing feedback..."):
                    result = analyze_single_feedback(feedback_text, st.session_state.analyzer)
                    if result:
                        # Add to session state
                        st.session_state.analysis_results.append(result)
                        
                        # Display result
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Sentiment", result['sentiment'])
                        with col2:
                            st.metric("Confidence", f"{result['confidence']:.3f}")
                        with col3:
                            sentiment_emoji = {"Positive": "😊", "Neutral": "😐", "Negative": "😞"}
                            st.markdown(f"### {sentiment_emoji.get(result['sentiment'], '🤔')}")
                        
                        # Detailed scores
                        st.subheader("Detailed Scores")
                        scores_df = pd.DataFrame([result['scores']])
                        st.bar_chart(scores_df.T)
                        
                        st.success("Analysis complete! Check the results section below.")
            else:
                st.warning("Please enter some feedback text to analyze.")
    
    elif analysis_method == "Batch Analysis":
        st.header("📝 Batch Analysis")
        
        # Input form
        batch_text = st.text_area(
            "Enter multiple feedback texts (one per line):",
            placeholder="Enter multiple feedback texts, one per line:\nThe fest was amazing!\nCould be better organized.\nGreat workshops and speakers.",
            height=200
        )
        
        if st.button("📊 Analyze All", type="primary"):
            if batch_text.strip():
                # Split by lines and filter empty lines
                feedback_lines = [line.strip() for line in batch_text.split('\n') if line.strip()]
                
                if feedback_lines:
                    with st.spinner(f"Analyzing {len(feedback_lines)} feedback texts..."):
                        results = analyze_batch_feedback(feedback_lines, st.session_state.analyzer)
                        if results:
                            # Add to session state
                            st.session_state.analysis_results.extend(results)
                            st.success(f"Successfully analyzed {len(results)} feedback texts!")
                else:
                    st.warning("Please enter at least one feedback text.")
            else:
                st.warning("Please enter some feedback texts to analyze.")
    
    elif analysis_method == "CSV Upload":
        st.header("📁 CSV Upload")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload CSV file with feedback data:",
            type=['csv'],
            help="CSV should contain a 'Feedback' column with the feedback text."
        )
        
        if uploaded_file is not None:
            try:
                # Try multiple methods to read CSV
                df = None
                error_messages = []
                
                # Method 1: Standard with quotes
                try:
                    df = pd.read_csv(uploaded_file, quotechar='"', skipinitialspace=True)
                except Exception as e:
                    error_messages.append(f"Method 1 failed: {str(e)}")
                
                # Method 2: Skip bad lines
                if df is None:
                    try:
                        uploaded_file.seek(0)  # Reset file pointer
                        df = pd.read_csv(uploaded_file, quotechar='"', skipinitialspace=True, on_bad_lines='skip')
                        if len(error_messages) > 0:
                            st.warning("Some lines were skipped due to parsing issues, but the file was loaded successfully.")
                    except Exception as e:
                        error_messages.append(f"Method 2 failed: {str(e)}")
                
                # Method 3: Different separator handling
                if df is None:
                    try:
                        uploaded_file.seek(0)  # Reset file pointer
                        df = pd.read_csv(uploaded_file, sep=',', quotechar='"', escapechar='\\', on_bad_lines='skip')
                    except Exception as e:
                        error_messages.append(f"Method 3 failed: {str(e)}")
                
                if df is None:
                    st.error("Could not parse CSV file. Please ensure it's properly formatted.")
                    with st.expander("Error details"):
                        for msg in error_messages:
                            st.text(msg)
                    return
                
                # Display preview
                st.subheader("📋 Data Preview")
                st.dataframe(df.head(), use_container_width=True)
                
                # Check for Feedback column
                if 'Feedback' not in df.columns:
                    st.error("CSV must contain a 'Feedback' column. Available columns: " + ", ".join(df.columns.tolist()))
                else:
                    # Show analysis button
                    if st.button("🚀 Upload & Analyze", type="primary"):
                        # Filter out empty feedback
                        df_clean = df[df['Feedback'].notna() & (df['Feedback'].str.strip() != '')]
                        
                        if len(df_clean) == 0:
                            st.warning("No valid feedback found in the CSV file.")
                        else:
                            with st.spinner(f"Analyzing {len(df_clean)} feedback entries..."):
                                try:
                                    results_df = st.session_state.analyzer.analyze_csv_feedback(df_clean)
                                    
                                    # Convert to list of dicts and add to session state
                                    results = results_df.to_dict('records')
                                    st.session_state.analysis_results.extend(results)
                                    
                                    st.success(f"Successfully analyzed {len(results)} feedback entries from CSV!")
                                    
                                except Exception as e:
                                    st.error(f"Error processing CSV: {str(e)}")
            
            except Exception as e:
                st.error(f"Error reading CSV file: {str(e)}")
    
    # Results section
    if st.session_state.analysis_results:
        st.markdown("---")
        st.header("📊 Analysis Results")
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["📈 Sentiment Distribution", "🔤 Word Analysis", "📋 Individual Results"])
        
        with tab1:
            create_sentiment_chart(st.session_state.analysis_results)
        
        with tab2:
            display_word_analysis(st.session_state.analysis_results, st.session_state.analyzer)
        
        with tab3:
            display_individual_results(st.session_state.analysis_results)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**College Event Feedback Analyzer** | Built with Streamlit and Rule-Based Sentiment Analysis"
    )

if __name__ == "__main__":
    main()