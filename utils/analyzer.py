from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import os
from huggingface_hub import login
from functools import lru_cache
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
MODEL_PATH = "./model_cache"
LABELS = ['Negative 😠', 'Neutral 😐', 'Positive 😊']

os.makedirs(MODEL_PATH, exist_ok=True)

@lru_cache(maxsize=1)
def load_model():
    """Load model with cache and fallback options"""
    try:
        logger.info("Attempting to download model...")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            cache_dir=MODEL_PATH
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            cache_dir=MODEL_PATH
        )
        logger.info("Model loaded successfully")
        return tokenizer, model
        
    except Exception as e:
        logger.warning(f"Online download failed: {e}")
        try:
            logger.info("Attempting to load cached model...")
            tokenizer = AutoTokenizer.from_pretrained(
                MODEL_PATH,
                local_files_only=True
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_PATH,
                local_files_only=True
            )
            logger.info("Cached model loaded successfully")
            return tokenizer, model
            
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise RuntimeError(
                "Could not load model. Please check your internet connection "
                "and ensure the model is cached."
            )

tokenizer, model = load_model()

def analyze_sentiment(text):
    """Analyze text sentiment with robust error handling"""
    if not isinstance(text, str) or not text.strip():
        return [0.33, 0.33, 0.33], "Neutral 😐"
    
    try:
        encoded_input = tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding=True
        )
        
        with torch.no_grad():
            output = model(**encoded_input)
        
        # Process scores
        scores = torch.nn.functional.softmax(
            output.logits,
            dim=1
        ).detach().numpy()[0]
        
        sentiment = LABELS[np.argmax(scores)]
        return scores, sentiment
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return [0.33, 0.33, 0.33], "Neutral 😐"