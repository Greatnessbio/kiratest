import os
import pandas as pd
from textstat import textstat
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import streamlit as st
import nltk

# Function to download NLTK resources
def download_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('sentiment/vader_lexicon')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('punkt')
        nltk.download('vader_lexicon')
        nltk.download('stopwords')

# Download NLTK resources
download_nltk_resources()

# Now initialize the SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Function to analyze a single row of text
def analyze_text(text, cta_words, salesy_words, newsy_words, custom_stopwords, sia):
    # Tokenize and remove stopwords
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha() and word not in custom_stopwords]
    
    # Flesch-Kincaid score
    fk_score = textstat.flesch_kincaid_grade(text)
    
    # Lexical diversity
    lexical_diversity = len(set(tokens)) / len(tokens) if tokens else 0
    
    # Top-performing words
    word_counts = Counter(tokens)
    top_words = word_counts.most_common(10)
    
    # Top-performing CTA words
    cta_counts = {word: tokens.count(word) for word in cta_words}
    
    # Sentiment analysis
    sentiment = sia.polarity_scores(text)
    
    # Sales-y vs News-y words
    salesy_count = sum(tokens.count(word) for word in salesy_words)
    newsy_count = sum(tokens.count(word) for word in newsy_words)
    
    return {
        'Flesch-Kincaid Score': fk_score,
        'Lexical Diversity': lexical_diversity,
        'Top Words': top_words,
        'CTA Words': cta_counts,
        'Sentiment': sentiment,
        'Sales-y Count': salesy_count,
        'News-y Count': newsy_count
    }

# Main function to load data and perform analysis
def main():
    st.title("Text Analysis Tool")
    
    # File
