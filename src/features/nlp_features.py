"""
NLP features for sentiment analysis and news processing.
Uses FinBERT and other language models for financial text analysis.
"""

import logging
from typing import Dict, List, Any, Optional
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


class NLPFeatures:
    """NLP features calculator for financial text analysis."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.tokenizer = None
        self.model = None
        self.initialized = False
    
    async def initialize(self):
        """Initialize NLP models."""
        try:
            # Load FinBERT model for financial sentiment analysis
            model_name = "ProsusAI/finbert"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            self.initialized = True
            self.logger.info("NLP features initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing NLP features: {e}")
            self.initialized = False
    
    async def calculate_sentiment_score(self, text: str) -> float:
        """Calculate sentiment score for financial text."""
        if not self.initialized or not text:
            return 0.0
        
        try:
            # Tokenize and encode text
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            
            # Get model prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Extract sentiment scores (positive, negative, neutral)
            sentiment_scores = predictions[0].numpy()
            
            # Calculate weighted sentiment score
            # positive: 1, neutral: 0, negative: -1
            sentiment_score = (sentiment_scores[0] * 1 + 
                             sentiment_scores[1] * 0 + 
                             sentiment_scores[2] * -1)
            
            return float(sentiment_score)
            
        except Exception as e:
            self.logger.error(f"Error calculating sentiment score: {e}")
            return 0.0
    
    async def extract_named_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities from financial text."""
        # Simplified implementation - in practice, you'd use spaCy or similar
        entities = []
        
        # Common financial entities
        financial_keywords = [
            'bitcoin', 'btc', 'ethereum', 'eth', 'crypto', 'cryptocurrency',
            'trading', 'market', 'price', 'bull', 'bear', 'pump', 'dump'
        ]
        
        text_lower = text.lower()
        for keyword in financial_keywords:
            if keyword in text_lower:
                entities.append({
                    'text': keyword,
                    'label': 'FINANCIAL_TERM',
                    'start': text_lower.find(keyword),
                    'end': text_lower.find(keyword) + len(keyword)
                })
        
        return entities
    
    async def calculate_topic_sentiment(self, texts: List[str]) -> Dict[str, float]:
        """Calculate sentiment for different topics."""
        if not texts:
            return {}
        
        topic_sentiments = {}
        
        # Group texts by topic (simplified)
        topics = {
            'price_movement': [],
            'market_sentiment': [],
            'news_events': []
        }
        
        for text in texts:
            text_lower = text.lower()
            if any(word in text_lower for word in ['price', 'up', 'down', 'rise', 'fall']):
                topics['price_movement'].append(text)
            elif any(word in text_lower for word in ['market', 'bull', 'bear', 'sentiment']):
                topics['market_sentiment'].append(text)
            else:
                topics['news_events'].append(text)
        
        # Calculate sentiment for each topic
        for topic, topic_texts in topics.items():
            if topic_texts:
                sentiments = []
                for text in topic_texts:
                    sentiment = await self.calculate_sentiment_score(text)
                    sentiments.append(sentiment)
                
                topic_sentiments[f'{topic}_sentiment'] = np.mean(sentiments)
                topic_sentiments[f'{topic}_count'] = len(topic_texts)
        
        return topic_sentiments
    
    async def calculate_news_impact_score(self, news_items: List[Dict[str, Any]]) -> float:
        """Calculate news impact score based on sentiment and volume."""
        if not news_items:
            return 0.0
        
        try:
            total_impact = 0.0
            total_weight = 0.0
            
            for item in news_items:
                text = item.get('title', '') + ' ' + item.get('content', '')
                sentiment = await self.calculate_sentiment_score(text)
                
                # Weight by source credibility and recency
                source_weight = item.get('source_credibility', 1.0)
                recency_weight = item.get('recency_weight', 1.0)
                weight = source_weight * recency_weight
                
                total_impact += sentiment * weight
                total_weight += weight
            
            return total_impact / total_weight if total_weight > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating news impact score: {e}")
            return 0.0
