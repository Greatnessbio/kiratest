import pandas as pd
from textstat import textstat
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import streamlit as st
import nltk

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
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Data Preview:", df.head())
        
        # Column selection
        text_column = st.selectbox("Select the column containing text data", df.columns)
        
        # User input for words
        cta_words = st.text_input("Enter CTA words separated by commas", "buy,subscribe,join,sign up,download").split(',')
        salesy_words = st.text_input("Enter sales-y words separated by commas", "deal,offer,discount,exclusive,limited").split(',')
        newsy_words = st.text_input("Enter news-y words separated by commas", "report,update,news,announcement,release").split(',')
        custom_stopwords = st.text_input("Enter additional stop words separated by commas", "").split(',')
        
        try:
            # Combine custom stopwords with default NLTK stopwords
            custom_stopwords = set(custom_stopwords) | set(stopwords.words('english'))
        except LookupError:
            st.error("NLTK stopwords data not found. Please ensure the NLTK data is downloaded and available.")
            return
        
        # Initialize sentiment analyzer
        sia = SentimentIntensityAnalyzer()
        
        # Apply the analysis to each row
        df['Analysis'] = df[text_column].apply(lambda text: analyze_text(text, cta_words, salesy_words, newsy_words, custom_stopwords, sia))
        
        # Expand the analysis dictionary into separate columns
        analysis_df = df['Analysis'].apply(pd.Series)
        result_df = pd.concat([df, analysis_df], axis=1).drop(columns=['Analysis'])
        
        # Display the result
        st.write("Analysis Result:", result_df.head())
        
        # Download button for the result
        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download analysis as CSV",
            data=csv,
            file_name='analyzed_data.csv',
            mime='text/csv',
        )

# Run the main function
if __name__ == "__main__":
    main()
