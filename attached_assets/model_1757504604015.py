import pandas as pd  # pyright: ignore[reportMissingImports]
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from scipy.special import softmax
from sklearn.model_selection import train_test_split
from collections import Counter
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import string
import re
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Avoid downloading NLTK data at import time to keep reloads fast and reliable.
# Resources will be ensured on first model load via a lazy initializer with fallbacks.

class TemperatureScaling(nn.Module):
    """Temperature scaling for confidence calibration"""
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits):
        return logits / self.temperature

class SentimentAnalyzer:
    """Sentiment analysis model for college event feedback"""
    
    def __init__(self):
        self.model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
        self.tokenizer = None
        self.model = None
        self.classifier = None
        self.temp_scale_model = None
        self.learned_temperature = None
        # NLTK resources are initialized lazily to avoid reload hangs
        self._nltk_ready = False
        self.english_stopwords = set()
        self.extra_stopwords = {
            'event','events','college','fest','fests','very','really','lot','much','many','also',
            'would','could','should','might','one','two','get','got','make','made','give','gave',
            'take','took','say','said','use','used','like','liked','okay','ok','please'
        }
        self.lemmatizer = None

        # Manual Hindi stopwords for Hinglish text
        self.manual_hindi_stopwords = [
            'tha', 'aur', 'hua', 'thi', 'par', 'kuch', 'nahi', 'ki', 'ka', 'ke',
            'mein', 'yeh', 'woh', 'hai', 'ho', 'bhi', 'liye', 'jaise', 'tab',
            'tak', 'fir', 'aa', 'gaya', 'ko', 'se', 'ne', 'le', 'de', 'kar'
        ]

        self.all_stopwords = set()

    def _ensure_nltk(self) -> None:
        """Ensure required NLTK resources exist; run once per process with graceful fallbacks."""
        if self._nltk_ready:
            return
        def _have_or_download(path: str, name: str) -> bool:
            try:
                nltk.data.find(path)
                return True
            except LookupError:
                try:
                    nltk.download(name, quiet=True)
                    nltk.data.find(path)
                    return True
                except Exception:
                    return False
        has_stop = _have_or_download('corpora/stopwords', 'stopwords')
        _have_or_download('tokenizers/punkt', 'punkt')
        has_wordnet = _have_or_download('corpora/wordnet', 'wordnet')
        _have_or_download('corpora/omw-1.4', 'omw-1.4')
        _ = (
            _have_or_download('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger') or
            _have_or_download('taggers/averaged_perceptron_tagger_eng', 'averaged_perceptron_tagger_eng')
        )

        try:
            self.english_stopwords = set(nltk.corpus.stopwords.words('english')) if has_stop else set()
        except Exception:
            self.english_stopwords = set()
        self.all_stopwords = self.english_stopwords.union(self.extra_stopwords).union(set(self.manual_hindi_stopwords))

        try:
            self.lemmatizer = WordNetLemmatizer() if has_wordnet else None
        except Exception:
            self.lemmatizer = None
        
        self._nltk_ready = True

    def _map_pos_to_wordnet(self, pos_tag: str):
        if pos_tag.startswith('J'):
            return wordnet.ADJ
        if pos_tag.startswith('V'):
            return wordnet.VERB
        if pos_tag.startswith('N'):
            return wordnet.NOUN
        if pos_tag.startswith('R'):
            return wordnet.ADV
        return wordnet.NOUN
        
    def load_model(self):
        """Load the pre-trained sentiment analysis model"""
        try:
            # Prepare NLTK resources once, without blocking reloads
            self._ensure_nltk()
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.classifier = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer)
            print("Model loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for better analysis"""
        if not isinstance(text, str):
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        return text
    
    def get_sentiment_scores(self, text: str) -> Dict[str, float]:
        """Get sentiment scores for a single text (3-class model support)"""
        if not self.model or not self.tokenizer:
            raise ValueError("Model not loaded. Call load_model() first.")

        text = self.preprocess_text(text)
        if not text:
            return {'Negative': 0.33, 'Neutral': 0.34, 'Positive': 0.33}

        try:
            encoded_text = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
            with torch.no_grad():
                output = self.model(**encoded_text)
            scores = output[0][0].detach().numpy()
            scores = softmax(scores)
            # For cardiffnlp/twitter-xlm-roberta-base-sentiment: [Negative, Neutral, Positive]
            scores_dict = {
                'Negative': float(scores[0]),
                'Neutral': float(scores[1]),
                'Positive': float(scores[2])
            }
            return scores_dict
        except Exception as e:
            print(f"Error processing text: {e}")
            return {'Negative': 0.33, 'Neutral': 0.34, 'Positive': 0.33}
    
    def get_sentiment_label(self, scores: Dict[str, float]) -> str:
        """Get the sentiment label from scores"""
        return max(scores, key=scores.get)
    
    def analyze_single_feedback(self, text: str) -> Dict:
        """Analyze a single feedback text"""
        scores = self.get_sentiment_scores(text)
        sentiment = self.get_sentiment_label(scores)
        confidence = max(scores.values())
        
        return {
            'text': text,
            'sentiment': sentiment,
            'confidence': confidence,
            'scores': scores
        }
    
    def analyze_batch_feedback(self, texts: List[str]) -> List[Dict]:
        """Analyze multiple feedback texts"""
        results = []
        for text in texts:
            result = self.analyze_single_feedback(text)
            results.append(result)
        return results
    
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
    
    def get_sentiment_distribution(self, results_df: pd.DataFrame) -> Dict[str, int]:
        """Get sentiment distribution from results"""
        return results_df['sentiment'].value_counts().to_dict()
    
    def get_most_common_words(self, texts: List[str], num_words: int = 10) -> List[Tuple[str, int]]:
        """Get most common meaningful words from a list of texts.
        Applies: lowercasing, punctuation removal, stopword removal, POS-tagging,
        lemmatization, and keeps alphabetic tokens of length >=3.
        """
        # Ensure stopwords/lemmatizer present (no-op after first time)
        self._ensure_nltk()
        all_words: List[str] = []
        for text in texts:
            if not isinstance(text, str):
                continue
            # Normalize spacing and lowercase
            clean_text = re.sub(r'\s+', ' ', text).strip().lower()
            try:
                tokens = nltk.word_tokenize(clean_text)
            except LookupError:
                # Fallback simple regex tokenization
                tokens = re.findall(r'[a-zA-Z]+', clean_text)
            # POS tagging for better lemmatization (fallback to nouns)
            try:
                pos_tags = nltk.pos_tag(tokens)
            except LookupError:
                pos_tags = [(tok, 'N') for tok in tokens]
            for token, pos in pos_tags:
                if token in string.punctuation:
                    continue
                if token in self.all_stopwords:
                    continue
                if not token.isalpha():
                    continue
                if len(token) < 3:
                    continue
                lemma = token
                if self.lemmatizer is not None:
                    try:
                        wn_pos = self._map_pos_to_wordnet(pos)
                        lemma = self.lemmatizer.lemmatize(token, wn_pos)
                    except Exception:
                        lemma = token
                if lemma in self.all_stopwords:
                    continue
                all_words.append(lemma)
        word_counts = Counter(all_words)
        return word_counts.most_common(num_words)
    
    def get_sentiment_words(self, results_df: pd.DataFrame, sentiment: str, num_words: int = 10) -> List[Tuple[str, int]]:
        """Get most common words for a specific sentiment"""
        sentiment_texts = results_df[results_df['sentiment'] == sentiment]['text'].tolist()
        return self.get_most_common_words(sentiment_texts, num_words)
    
    def calibrate_confidence(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Apply temperature scaling for confidence calibration"""
        if len(results_df) < 20:  # Need minimum data for calibration
            return results_df
        
        try:
            # Split data for calibration
            train_df, val_df = train_test_split(results_df, test_size=0.2, random_state=42)
            
            # Get logits for validation set
            val_logits = []
            for text in val_df['text']:
                encoded_text = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
                with torch.no_grad():
                    output = self.model(**encoded_text)
                val_logits.append(output.logits.squeeze().numpy())
            
            val_logits = np.array(val_logits)
            
            # Convert to tensors
            val_logits_tensor = torch.from_numpy(val_logits)
            # Map sentiment labels to 5-class model outputs
            sentiment_to_label = {'Negative': 0, 'Neutral': 2, 'Positive': 4}  # Map to 1-star, 3-star, 5-star
            val_labels_tensor = torch.tensor([sentiment_to_label[s] for s in val_df['sentiment'].values])
            
            # Initialize temperature scaling
            temp_scale_model = TemperatureScaling()
            optimizer = torch.optim.LBFGS([temp_scale_model.temperature], lr=0.01, max_iter=50)
            criterion = nn.CrossEntropyLoss()
            
            # Optimize temperature
            def eval():
                loss = criterion(temp_scale_model(val_logits_tensor), val_labels_tensor)
                loss.backward()
                return loss
            
            optimizer.step(eval)
            learned_temperature = temp_scale_model.temperature.item()
            
            # Apply calibration to all data
            all_logits = []
            for text in results_df['text']:
                encoded_text = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
                with torch.no_grad():
                    output = self.model(**encoded_text)
                all_logits.append(output.logits.squeeze().numpy())
            
            all_logits = np.array(all_logits)
            all_logits_tensor = torch.from_numpy(all_logits)
            calibrated_logits = temp_scale_model(all_logits_tensor)
            calibrated_probs = torch.softmax(calibrated_logits, dim=1).detach().numpy()
            
            # Update results with calibrated confidence
            results_df = results_df.copy()
            results_df['calibrated_confidence'] = np.max(calibrated_probs, axis=1)
            
            return results_df
            
        except Exception as e:
            print(f"Error in confidence calibration: {e}")
            return results_df

# Global instance
sentiment_analyzer = SentimentAnalyzer()
