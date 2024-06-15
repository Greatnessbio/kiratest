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

# Set the NLTK data path to the local directory
nltk.data.path.append(os.path.join(os.path.dirname(__file__), 'nltk_data'))

# Function to download NLTK resources
def download_nltk_resources():
    nltk_resources = ['tokenizers/punkt', 'sentiment/vader_lexicon', 'corpora/stopwords']
    for resource in nltk_resources:
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(resource.split('/')[-1])

# Download NLTK resources
download_nltk_resources()

# Now initialize the SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Function to analyze a single row of text
def analyze_text(text, cta_words, salesy_words, newsy_words, custom_stopwords, sia):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha() and word not in custom_stopwords]
    
    fk_score = textstat.flesch_kincaid_grade(text)
    lexical_diversity = len(set(tokens)) / len(tokens) if tokens else 0
    word_counts = Counter(tokens)
    top_words = word_counts.most_common(10)
    cta_counts = {word: tokens.count(word) for word in cta_words}
    sentiment = sia.polarity_scores(text)
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
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None
