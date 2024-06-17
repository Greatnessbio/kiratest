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
    
    # Initialize Sentiment Analyzer
    analyzer = SentimentIntensityAnalyzer()
    
    # Create a new DataFrame to store the results
    results = pd.DataFrame(columns=[
        'Row', 'Flesch-Kincaid Score', 'Lexical Diversity', 'Top Words', 
        'Sentiment', 'Top CTA Words', 'Sales-y Words Count', 'News-y Words Count'
    ])
    
    # Define CTA, sales-y, and news-y words
    cta_words = ['buy', 'call', 'subscribe', 'join', 'register', 'order', 'book', 'shop', 'get', 'reserve']
    salesy_words = ['buy', 'discount', 'offer', 'sale', 'deal', 'price', 'save', 'free', 'limited', 'exclusive']
    newsy_words = ['report', 'news', 'update', 'announce', 'release', 'statement', 'coverage', 'headline', 'breaking', 'story']
    
    # Process each row
    for index, row in df.iterrows():
        text_data = str(row[column])
        
        # Flesch-Kincaid score
        flesch_kincaid_score = textstat.flesch_kincaid_grade(text_data)
        
        # Lexical diversity
        words = nltk.word_tokenize(text_data)
        lexical_diversity = len(set(words)) / len(words) if words else 0
        
        # Top-performing words
        fdist = FreqDist(words)
        top_words = fdist.most_common(10)
        
        # Sentiment analysis
        sentiment = analyzer.polarity_scores(text_data)
        
        # Top-performing CTA words
        cta_word_counts = {word: words.count(word) for word in cta_words}
        
        # "Sales-y" vs "News-y" words
        salesy_count = sum(words.count(word) for word in salesy_words)
        newsy_count = sum(words.count(word) for word in newsy_words)
        
        # Append results to the DataFrame
        results = results.append({
            'Row': index,
            'Flesch-Kincaid Score': flesch_kincaid_score,
            'Lexical Diversity': lexical_diversity,
            'Top Words': top_words,
            'Sentiment': sentiment,
            'Top CTA Words': cta_word_counts,
            'Sales-y Words Count': salesy_count,
            'News-y Words Count': newsy_count
        }, ignore_index=True)
    
    # Display the results
    st.write('Analysis Results:')
    st.write(results)
    
    # Provide explanations
    st.write('### Explanations:')
    st.write('**Flesch-Kincaid Score:** A readability test designed to indicate how difficult a passage in English is to understand.')
    st.write('**Lexical Diversity:** A measure of how many different words are used in the text.')
    st.write('**Top Words:** The most frequently occurring words in the text.')
    st.write('**Sentiment Analysis:** An assessment of the emotional tone of the text.')
    st.write('**Top CTA Words:** The most frequently occurring call-to-action words in the text.')
    st.write('**Sales-y Words Count:** The number of words in the text that are typically associated with sales language.')
    st.write('**News-y Words Count:** The number of words in the text that are typically associated with news language.')

