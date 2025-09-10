import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

st.title("My Streamlit App")
st.write("Hello, world!")
# Try to import ML libraries - fallback to rule-based if not available
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Advanced Hinglish Feedback Analyzer",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

class AdvancedHinglishSentimentAnalyzer:
    """Advanced sentiment analyzer with ML and rule-based fallback"""
    
    def __init__(self):
        self.ml_model = None
        self.tokenizer = None
        self.use_ml = ML_AVAILABLE
        
        # Enhanced Hinglish word dictionaries
        self.positive_words = {
            # English positive words
            'amazing', 'excellent', 'great', 'awesome', 'fantastic', 'wonderful', 
            'brilliant', 'outstanding', 'perfect', 'good', 'best', 'love', 
            'enjoy', 'helpful', 'organized', 'interactive', 'knowledgeable',
            'memorable', 'fun', 'interesting', 'superb', 'marvelous', 'fabulous',
            'nice', 'cool', 'sweet', 'beautiful', 'lovely', 'incredible',
            
            # Hindi/Hinglish positive words
            'accha', 'achha', 'badhiya', 'bahut', 'zabardast', 'kamaal', 'mast',
            'shandaar', 'shandar', 'ekdam', 'top', 'maja', 'mazaa', 'maza', 
            'dhansu', 'dhaasu', 'jhakaas', 'jhakkas', 'sundar', 'khubsurat',
            'lajawab', 'umda', 'behtarin', 'shaandaar', 'gazab', 'solid',
            'bindaas', 'jordar', 'zordaar', 'shandar', 'khushi', 'anand',
            'prasann', 'santusht', 'dilkush', 'dil', 'heart', 'pyaar', 'mohabbat',
            'ishq', 'pasand', 'favorite', 'favourite', 'best', 'number', 'one',
            'first', 'top', 'class', 'quality', 'level', 'next', 'badiya',
            'shandar', 'lajawab', 'behtreen', 'umda', 'khas', 'special'
        }
        
        self.negative_words = {
            # English negative words
            'bad', 'terrible', 'awful', 'boring', 'poor', 'disappointed', 
            'waste', 'worst', 'hate', 'useless', 'delayed', 'disorganized',
            'lacking', 'mediocre', 'average', 'pathetic', 'disgusting',
            'horrible', 'annoying', 'frustrating', 'sad', 'angry', 'upset',
            
            # Hindi/Hinglish negative words
            'bekar', 'bekaar', 'ganda', 'bura', 'kharab', 'kharrab', 'faltu',
            'bakwas', 'bewakoof', 'pagal', 'stupid', 'nonsense', 'timepass',
            'boring', 'dull', 'sad', 'upset', 'angry', 'gussa', 'ghussa',
            'pareshan', 'tension', 'problem', 'issue', 'dikkat', 'mushkil',
            'galat', 'wrong', 'mistake', 'ghalti', 'nirasha', 'udaas', 
            'dukh', 'dard', 'pain', 'hurt', 'thak', 'thaka', 'tired', 
            'bore', 'nautanki', 'drama', 'fake', 'jhooth', 'lie', 'crap',
            'shit', 'bullshit', 'nonsense', 'rubbish', 'garbage', 'trash',
            'pathetic', 'disgusting', 'horrible', 'terrible', 'awful'
        }
        
        # Intensity modifiers
        self.intensifiers = {
            'bahut', 'very', 'really', 'so', 'too', 'extremely', 'super', 
            'highly', 'totally', 'completely', 'absolutely', 'quite',
            'bilkul', 'ekdam', 'poora', 'pura', 'full', 'zara', 'thoda'
        }
        
        self.diminishers = {
            'little', 'bit', 'slightly', 'somewhat', 'kind', 'sort', 'of',
            'thoda', 'zara', 'kam', 'less', 'not', 'so', 'much'
        }
        
        if self.use_ml:
            self._load_ml_model()
    
    def _load_ml_model(self):
        """Load Hugging Face model for better accuracy"""
        try:
            # Try different Hinglish/multilingual models in order of preference
            models_to_try = [
                "cardiffnlp/twitter-xlm-roberta-base-sentiment",  # Multilingual
                "nlptown/bert-base-multilingual-uncased-sentiment",  # Multilingual
                "distilbert-base-uncased"  # Fallback English model
            ]
            
            for model_name in models_to_try:
                try:
                    self.ml_model = pipeline(
                        "sentiment-analysis", 
                        model=model_name,
                        return_all_scores=True
                    )
                    break
                except Exception as e:
                    continue
            
            if self.ml_model is None:
                self.use_ml = False
                
        except Exception as e:
            self.use_ml = False
    
    def preprocess_text(self, text: str) -> str:
        """Enhanced text preprocessing for Hinglish"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Handle common Hinglish patterns and normalize spellings
        replacements = {
            'accha': 'achha', 'acha': 'achha', 'achchha': 'achha',
            'badia': 'badhiya', 'badiya': 'badhiya', 'badya': 'badhiya',
            'kharab': 'kharrab', 'khrab': 'kharrab',
            'jakas': 'jhakaas', 'jhakass': 'jhakaas', 'jhakas': 'jhakaas',
            'dhansu': 'dhaasu', 'dhasu': 'dhaasu',
            'shandar': 'shandaar', 'shanda': 'shandaar',
            'kammal': 'kamaal', 'kamal': 'kamaal',
            'gajab': 'gazab', 'gzab': 'gazab',
            'bekaar': 'bekar', 'bkar': 'bekar',
            'bura': 'bura', 'bra': 'bura',
            'ganda': 'ganda', 'gnda': 'ganda',
            'gussa': 'ghussa', 'gusa': 'ghussa'
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Remove extra whitespace and special characters
        text = re.sub(r'[^\w\s]', ' ', text)
        return re.sub(r'\s+', ' ', text.strip())
    
    def analyze_single_feedback_ml(self, text: str) -> Dict:
        """Analyze using ML model"""
        try:
            results = self.ml_model(text)
            
            # Convert model output to our format
            if isinstance(results[0], list):
                # Model returns all scores
                scores_dict = {}
                for item in results[0]:
                    label = item['label'].upper()
                    if 'POS' in label or label == 'POSITIVE':
                        scores_dict['Positive'] = item['score']
                    elif 'NEG' in label or label == 'NEGATIVE':
                        scores_dict['Negative'] = item['score']
                    elif 'NEU' in label or label == 'NEUTRAL':
                        scores_dict['Neutral'] = item['score']
            else:
                # Single prediction
                label = results[0]['label'].upper()
                score = results[0]['score']
                
                if 'POS' in label or label == 'POSITIVE':
                    scores_dict = {'Positive': score, 'Negative': (1-score)/2, 'Neutral': (1-score)/2}
                elif 'NEG' in label or label == 'NEGATIVE':
                    scores_dict = {'Negative': score, 'Positive': (1-score)/2, 'Neutral': (1-score)/2}
                else:
                    scores_dict = {'Neutral': score, 'Positive': (1-score)/2, 'Negative': (1-score)/2}
            
            # Ensure all three sentiments are present
            for sentiment in ['Positive', 'Negative', 'Neutral']:
                if sentiment not in scores_dict:
                    scores_dict[sentiment] = 0.0
            
            # Normalize scores to sum to 1
            total = sum(scores_dict.values())
            if total > 0:
                scores_dict = {k: v/total for k, v in scores_dict.items()}
            
            sentiment = max(scores_dict, key=scores_dict.get)
            confidence = max(scores_dict.values())
            
            return {
                'text': text,
                'sentiment': sentiment,
                'confidence': confidence,
                'scores': scores_dict,
                'method': 'ML'
            }
            
        except Exception as e:
            return self.analyze_single_feedback_rule_based(text)
    
    def analyze_single_feedback_rule_based(self, text: str) -> Dict:
        """Enhanced rule-based analysis with intensity handling"""
        processed_text = self.preprocess_text(text)
        words = processed_text.split()
        
        positive_score = 0
        negative_score = 0
        intensity_multiplier = 1.0
        
        for i, word in enumerate(words):
            # Check for intensity modifiers
            if word in self.intensifiers:
                intensity_multiplier = 1.5
                continue
            elif word in self.diminishers:
                intensity_multiplier = 0.7
                continue
            
            # Score sentiment words
            if word in self.positive_words:
                positive_score += intensity_multiplier
            elif word in self.negative_words:
                negative_score += intensity_multiplier
            
            # Reset intensity for next word
            intensity_multiplier = 1.0
        
        # Determine sentiment with more nuanced scoring
        total_score = positive_score + negative_score
        if total_score == 0:
            sentiment = 'Neutral'
            confidence = 0.5
            scores = {'Positive': 0.33, 'Negative': 0.33, 'Neutral': 0.34}
        else:
            pos_ratio = positive_score / total_score
            neg_ratio = negative_score / total_score
            
            if pos_ratio > neg_ratio:
                sentiment = 'Positive'
                confidence = min(0.95, 0.5 + pos_ratio * 0.45)
                scores = {
                    'Positive': confidence,
                    'Negative': (1 - confidence) * neg_ratio / (pos_ratio + neg_ratio) if (pos_ratio + neg_ratio) > 0 else (1 - confidence) / 2,
                    'Neutral': (1 - confidence) * (1 - neg_ratio / (pos_ratio + neg_ratio)) if (pos_ratio + neg_ratio) > 0 else (1 - confidence) / 2
                }
            elif neg_ratio > pos_ratio:
                sentiment = 'Negative'
                confidence = min(0.95, 0.5 + neg_ratio * 0.45)
                scores = {
                    'Negative': confidence,
                    'Positive': (1 - confidence) * pos_ratio / (pos_ratio + neg_ratio) if (pos_ratio + neg_ratio) > 0 else (1 - confidence) / 2,
                    'Neutral': (1 - confidence) * (1 - pos_ratio / (pos_ratio + neg_ratio)) if (pos_ratio + neg_ratio) > 0 else (1 - confidence) / 2
                }
            else:
                sentiment = 'Neutral'
                confidence = 0.6
                scores = {'Positive': 0.2, 'Negative': 0.2, 'Neutral': 0.6}
        
        return {
            'text': text,
            'sentiment': sentiment,
            'confidence': confidence,
            'scores': scores,
            'method': 'Rule-based'
        }
    
    def analyze_single_feedback(self, text: str) -> Dict:
        """Main analysis method that chooses ML or rule-based"""
        if self.use_ml and self.ml_model:
            return self.analyze_single_feedback_ml(text)
        else:
            return self.analyze_single_feedback_rule_based(text)
    
    def analyze_batch_feedback(self, texts: List[str]) -> List[Dict]:
        """Analyze multiple feedback texts"""
        return [self.analyze_single_feedback(text) for text in texts]
    
    def analyze_csv_feedback(self, df: pd.DataFrame, text_column: str = 'Feedback') -> pd.DataFrame:
        """Analyze feedback from a CSV DataFrame"""
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame")
        
        results = []
        progress_bar = st.progress(0)
        total_rows = len(df)
        
        for idx, row in df.iterrows():
            text = str(row[text_column])
            result = self.analyze_single_feedback(text)
            results.append(result)
            progress_bar.progress((idx + 1) / total_rows)
        
        progress_bar.empty()
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Add individual score columns
        scores_df = pd.json_normalize(results_df['scores'])
        results_df = pd.concat([results_df.drop('scores', axis=1), scores_df], axis=1)
        
        return results_df
    
    def get_most_common_words(self, texts: List[str], num_words: int = 10) -> List[tuple]:
        """Get most common words from texts (English + Hinglish)"""
        all_words = []
        # Enhanced stopwords for English + Hindi/Hinglish
        stopwords = {
            # English stopwords
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 
            'a', 'an', 'is', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 
            'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 
            'can', 'it', 'he', 'she', 'they', 'we', 'you', 'i', 'me', 'my', 'your', 
            'his', 'her', 'their', 'our', 'this', 'that', 'these', 'those', 'are',
            
            # Hindi/Hinglish stopwords  
            'hai', 'tha', 'thi', 'the', 'aur', 'ya', 'ki', 'ka', 'ke', 'ko', 'se', 
            'mein', 'me', 'par', 'pe', 'main', 'yeh', 'ye', 'woh', 'wo', 'kya', 
            'kyun', 'kaise', 'kab', 'kahan', 'kitna', 'kaun', 'jo', 'jab', 'agar', 
            'lekin', 'par', 'bas', 'sirf', 'bhi', 'tak', 'liye', 'lie', 'gaya', 
            'gaye', 'gayi', 'kar', 'kiya', 'kiye', 'karna', 'hona', 'hone', 'hua', 
            'hui', 'hue', 'rahega', 'rahenge', 'rahe', 'raha', 'rahi', 'rhe', 'rha', 
            'rhi', 'aise', 'jaise', 'waise', 'kaise', 'very', 'much', 'more', 'most',
            'some', 'any', 'all', 'many', 'few', 'little', 'big', 'small', 'new', 'old'
        }
        
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
        with st.spinner("Initializing Advanced Hinglish Sentiment Analyzer..."):
            st.session_state.analyzer = AdvancedHinglishSentimentAnalyzer()

def clear_results():
    """Clear all analysis results"""
    st.session_state.analysis_results = []
    st.success("Results cleared successfully!")
    st.rerun()

def analyze_single_feedback(text: str, analyzer: AdvancedHinglishSentimentAnalyzer) -> Dict:
    """Analyze a single feedback text"""
    try:
        result = analyzer.analyze_single_feedback(text)
        return result
    except Exception as e:
        st.error(f"Error analyzing feedback: {str(e)}")
        return None

def analyze_batch_feedback(texts: List[str], analyzer: AdvancedHinglishSentimentAnalyzer) -> List[Dict]:
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
    method_counts = {}
    
    for result in results:
        sentiment = result['sentiment']
        sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
        
        method = result.get('method', 'Unknown')
        method_counts[method] = method_counts.get(method, 0) + 1
    
    if not sentiment_counts:
        st.info("No sentiment data available")
        return
    
    import matplotlib.pyplot as plt
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sentiment Distribution")
        chart_data = pd.DataFrame(
            list(sentiment_counts.items()),
            columns=['Sentiment', 'Count']
        )
        # Pie chart for sentiment
        fig, ax = plt.subplots()
        ax.pie(chart_data['Count'], labels=chart_data['Sentiment'], autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
        ax.axis('equal')
        st.pyplot(fig)
    
    with col2:
        st.subheader("Analysis Method Used")
        if method_counts:
            method_data = pd.DataFrame(
                list(method_counts.items()),
                columns=['Method', 'Count']
            )
            # Pie chart for method
            fig2, ax2 = plt.subplots()
            ax2.pie(method_data['Count'], labels=method_data['Method'], autopct='%1.1f%%', startangle=90, colors=plt.cm.Pastel1.colors)
            ax2.axis('equal')
            st.pyplot(fig2)
    
    # Display statistics
    total = sum(sentiment_counts.values())
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Analyzed", total)
    
    with col2:
        positive_count = sentiment_counts.get('Positive', 0)
        positive_pct = (positive_count / total * 100) if total > 0 else 0
        st.metric("Positive", f"{positive_count}", f"{positive_pct:.1f}%")
    
    with col3:
        neutral_count = sentiment_counts.get('Neutral', 0)
        neutral_pct = (neutral_count / total * 100) if total > 0 else 0
        st.metric("Neutral", f"{neutral_count}", f"{neutral_pct:.1f}%")
    
    with col4:
        negative_count = sentiment_counts.get('Negative', 0)
        negative_pct = (negative_count / total * 100) if total > 0 else 0
        st.metric("Negative", f"{negative_count}", f"{negative_pct:.1f}%")

def display_word_analysis(results: List[Dict], analyzer: AdvancedHinglishSentimentAnalyzer):
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
                # Pie chart for positive words
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots()
                ax.pie(pos_df['Count'], labels=pos_df['Word'], autopct='%1.1f%%', startangle=90, colors=plt.cm.Greens(np.linspace(0.5, 1, len(pos_df))))
                ax.axis('equal')
                st.pyplot(fig)
                st.write("Top words:", ", ".join([word for word, count in positive_words[:5]]))
            else:
                st.info("No positive words found")
        
        with col2:
            st.subheader("🔴 Most Common Negative Words")
            if negative_words:
                neg_df = pd.DataFrame(negative_words, columns=['Word', 'Count'])
                # Pie chart for negative words
                fig2, ax2 = plt.subplots()
                ax2.pie(neg_df['Count'], labels=neg_df['Word'], autopct='%1.1f%%', startangle=90, colors=plt.cm.Reds(np.linspace(0.5, 1, len(neg_df))))
                ax2.axis('equal')
                st.pyplot(fig2)
                st.write("Top words:", ", ".join([word for word, count in negative_words[:5]]))
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
            method = result.get('method', 'Unknown')
            scores = result.get('scores', {})
        else:
            # Handle DataFrame row format
            text = getattr(result, 'text', 'N/A')
            sentiment = getattr(result, 'sentiment', 'N/A')
            confidence = getattr(result, 'confidence', 0)
            method = getattr(result, 'method', 'Unknown')
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
            'Method': method,
            'Confidence': f"{confidence:.3f}" if isinstance(confidence, (int, float)) else str(confidence),
            'Positive': f"{scores.get('Positive', 0):.3f}",
            'Neutral': f"{scores.get('Neutral', 0):.3f}",
            'Negative': f"{scores.get('Negative', 0):.3f}"
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
    st.title("🎓 Advanced Hinglish Feedback Analyzer")
    #st.markdown("**Powered by AI**: Analyze sentiment of feedback from college events in English, Hindi, and Hinglish")
    
    # Add note about analyzer capabilities
    if ML_AVAILABLE:
        st.success("🤖 Using state-of-the-art machine learning models for superior accuracy in Hinglish sentiment analysis!")
    else:
        st.info("📚 **Enhanced Rule-based Mode**: Sophisticated analysis with extensive Hinglish vocabulary. Try 'Event zabardast tha!' or 'Bilkul faltu organization!'")
    
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
        method_counts = {}
        for result in st.session_state.analysis_results:
            sentiment = result['sentiment']
            sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
            method = result.get('method', 'Unknown')
            method_counts[method] = method_counts.get(method, 0) + 1
        
        for sentiment, count in sentiment_counts.items():
            st.sidebar.metric(sentiment, count)
        
        if method_counts:
            st.sidebar.markdown("**Analysis Methods:**")
            for method, count in method_counts.items():
                st.sidebar.text(f"{method}: {count}")
    
    # Main content area - same as before but with enhanced examples
    if analysis_method == "Single Feedback":
        st.header("💬 Single Feedback Analysis")
        
        # Input form
        feedback_text = st.text_area(
            "Enter your feedback (English/Hindi/Hinglish):",
            placeholder="Try: 'Event zabardast tha! Bahut maza aaya' or 'Bilkul bekar organization tha yaar'",
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
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Sentiment", result['sentiment'])
                        with col2:
                            st.metric("Confidence", f"{result['confidence']:.3f}")
                        with col3:
                            st.metric("Method", result.get('method', 'Unknown'))
                        with col4:
                            sentiment_emoji = {"Positive": "😊", "Neutral": "😐", "Negative": "😞"}
                            st.markdown(f"### {sentiment_emoji.get(result['sentiment'], '🤔')}")
                        
                        # Detailed scores
                        st.subheader("Detailed Scores")
                        scores_df = pd.DataFrame([result['scores']])
                        import matplotlib.pyplot as plt
                        fig, ax = plt.subplots()
                        ax.pie(scores_df.iloc[0], labels=scores_df.columns, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
                        ax.axis('equal')
                        st.pyplot(fig)
                        
                        st.success("Analysis complete! Check the results section below.")
            else:
                st.warning("Please enter some feedback text to analyze.")
    
    elif analysis_method == "Batch Analysis":
        st.header("📝 Batch Analysis")
        
        # Input form
        batch_text = st.text_area(
            "Enter multiple feedback texts (one per line):",
            placeholder="Example:\nEvent zabardast tha!\nBekar organization\nBahut maza aaya fest mein",
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
            help="CSV should contain a 'Feedback' column with the feedback text (English/Hindi/Hinglish)."
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
                                    
                                    # Show analysis summary
                                    sentiment_dist = results_df['sentiment'].value_counts()
                                    method_dist = results_df['method'].value_counts() if 'method' in results_df.columns else {}
                                    
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.subheader("Sentiment Summary")
                                        st.write(sentiment_dist)
                                    with col2:
                                        if method_dist.any():
                                            st.subheader("Method Summary")
                                            st.write(method_dist)
                                    
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
    footer_text = "**Advanced Hinglish Feedback Analyzer** | "
    if ML_AVAILABLE:
        footer_text += "Powered by Hugging Face Transformers & Advanced Rule-based Analysis"
    else:
        footer_text += "Enhanced Rule-based Sentiment Analysis with Hinglish Support"
    st.markdown(footer_text)

if __name__ == "__main__":
    main()
