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
    
    # User input for sales-y and news-y words
    salesy_words_input = st.text_area('Enter sales-y words (comma-separated)', 'buy,discount,offer,sale,deal,price,save,free,limited,exclusive')
    newsy_words_input = st.text_area('Enter news-y words (comma-separated)', 'report,news,update,announce,release,statement,coverage,headline,breaking,story')
    
    salesy_words = [word.strip() for word in salesy_words_input.split(',')]
    newsy_words = [word.strip() for word in newsy_words_input.split(',')]
    
    # Initialize Sentiment Analyzer
    analyzer = SentimentIntensityAnalyzer()
    
    # Create a new DataFrame to store the results
    results = pd.DataFrame(columns=[
        'Text', 'Flesch-Kincaid Score', 'Lexical Diversity', 'Top Words', 
        'Sentiment', 'Top CTA Words', 'Sales-y Words Count', 'News-y Words Count'
    ])
    
    # Define CTA words
    cta_words = ['buy', 'call', 'subscribe', 'join', 'register', 'order', 'book', 'shop', 'get', 'reserve']
    
    # Process each row
    for index, row in df.iterrows():
        text_data = str(row[column])
        
        # Flesch-Kincaid score
        flesch_kincaid_score = textstat.flesch_kincaid_grade(text_data)
        if flesch_kincaid_score > 12:
            flesch_kincaid_score = 12
        
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
        new_row = pd.DataFrame({
            'Text': [text_data],
            'Flesch-Kincaid Score': [flesch_kincaid_score],
            'Lexical Diversity': [lexical_diversity],
            'Top Words': [str(top_words)],
            'Sentiment': [str(sentiment)],
            'Top CTA Words': [str(cta_word_counts)],
            'Sales-y Words Count': [salesy_count],
            'News-y Words Count': [newsy_count]
        })
        results = pd.concat([results, new_row], ignore_index=True)
    
    # Display the results
    st.write('Analysis Results:')
    st.dataframe(results)
    
    # Provide explanations
    st.write('### Explanations:')
    st.write('**Flesch-Kincaid Score:** A readability test designed to indicate how difficult a passage in English is to understand. The score is typically between 0 and 12, with higher scores indicating more difficult text. A score of 8-10 is considered fairly difficult, while a score of 12 is very difficult.')
    st.write('**Lexical Diversity:** A measure of how many different words are used in the text. It is calculated as the ratio of unique words to the total number of words. A higher ratio indicates a more diverse vocabulary.')
    st.write('**Top Words:** The most frequently occurring words in the text. This helps identify common themes or topics.')
    st.write('**Sentiment Analysis:** An assessment of the emotional tone of the text. The sentiment score ranges from -1 (very negative) to 1 (very positive). A score close to 0 indicates neutral sentiment.')
    st.write('**Top CTA Words:** The most frequently occurring call-to-action words in the text. These words are often used to encourage readers to take specific actions.')
    st.write('**Sales-y Words Count:** The number of words in the text that are typically associated with sales language. A higher count indicates a more promotional tone.')
    st.write('**News-y Words Count:** The number of words in the text that are typically associated with news language. A higher count indicates a more informational tone.')
    
    # Display charts for better visualization
    st.write('### Charts:')
    st.bar_chart(results[['Flesch-Kincaid Score', 'Lexical Diversity', 'Sales-y Words Count', 'News-y Words Count']])
    st.line_chart(results[['Sentiment']])
