import streamlit as st
import pandas as pd
import textstat
import nltk
from nltk.probability import FreqDist
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

nltk.download('punkt')

st.title('CSV File Analysis')

# Upload CSV file
uploaded_file = st.file_uploader('Upload a CSV file', type='csv')

if uploaded_file is not None:
    # Read the CSV file
    df = pd.read_csv(uploaded_file)
    
    # Show preview of the CSV file
    st.write('Preview of the CSV file:')
    st.write(df.head())
    
    # Select column to process
    column = st.selectbox('Select column to process', df.columns)
    
    # Checkbox to indicate if the CSV has headers
    has_headers = st.checkbox('CSV has headers', value=True)
    
    # Process the selected column
    text_data = df[column].astype(str).str.cat(sep=' ')
    
    # Flesch-Kincaid score
    flesch_kincaid_score = textstat.flesch_kincaid_grade(text_data)
    st.write(f'Flesch-Kincaid Score: {flesch_kincaid_score}')
    
    # Lexical diversity
    words = nltk.word_tokenize(text_data)
    lexical_diversity = len(set(words)) / len(words)
    st.write(f'Lexical Diversity: {lexical_diversity}')
    
    # Top-performing words
    fdist = FreqDist(words)
    top_words = fdist.most_common(10)
    st.write('Top-performing words:')
    st.write(top_words)
    
    # Sentiment analysis
    analyzer = SentimentIntensityAnalyzer()
    sentiment = analyzer.polarity_scores(text_data)
    st.write('Sentiment Analysis:')
    st.write(sentiment)
    
    # Top-performing CTA words
    cta_words = ['buy', 'call', 'subscribe', 'join', 'register', 'order', 'book', 'shop', 'get', 'reserve']
    cta_word_counts = {word: words.count(word) for word in cta_words}
    st.write('Top-performing CTA words:')
    st.write(cta_word_counts)

    # "Sales-y" vs "News-y" words
    salesy_words = ['buy', 'discount', 'offer', 'sale', 'deal', 'price', 'save', 'free', 'limited', 'exclusive']
    newsy_words = ['report', 'news', 'update', 'announce', 'release', 'statement', 'coverage', 'headline', 'breaking', 'story']

    salesy_count = sum(words.count(word) for word in salesy_words)
    newsy_count = sum(words.count(word) for word in newsy_words)

    st.write('"Sales-y" vs "News-y" words:')
    st.write(f'Sales-y words count: {salesy_count}')
    st.write(f'News-y words count: {newsy_count}')
