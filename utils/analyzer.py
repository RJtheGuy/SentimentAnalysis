# utils/analyzer.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
LABELS = ['Negative 😠', 'Neutral 😐', 'Positive 😊']

# Load once globally
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

def analyze_sentiment(text):
    encoded_input = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    output = model(**encoded_input)
    scores = torch.nn.functional.softmax(output.logits, dim=1).detach().numpy()[0]
    sentiment = LABELS[np.argmax(scores)]
    return scores, sentiment
