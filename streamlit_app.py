import streamlit as st
import pandas as pd
import nltk
from nltk.probability import FreqDist
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import textstat

# Ensure necessary NLTK data is downloaded
nltk.download('punkt')

# Define functions for additional readability metrics
def new_dale_chall(text):
    return textstat.dale_chall_readability_score(text)

def smog_index(text):
    return textstat.smog_index(text)

# Add other readability metrics as needed

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
    
    # User input for sales-y and news-y words
    salesy_words_input = st.text_area('Enter sales-y words (comma-separated)', 'buy,discount,offer,sale,deal,price,save,free,limited,exclusive')
    newsy_words_input = st.text_area('Enter news-y words (comma-separated)', 'report,news,update,announce,release,statement,coverage,headline,breaking,story')
    
    salesy_words = [word.strip() for word in salesy_words_input.split(',')]
    newsy_words = [word.strip() for word in newsy_words_input.split(',')]
    
    # Prefilled and editable CTA words
    default_cta_words = 'buy,call,subscribe,join,register,order,book,shop,get,reserve'
    cta_words_input = st.text_area('Enter CTA words (comma-separated)', default_cta_words)
    cta_words = [word.strip() for word in cta_words_input.split(',')]
    
    # Generate Report button
    if st.button('Generate Report'):
        # Initialize Sentiment Analyzer
        analyzer = SentimentIntensityAnalyzer()
        
        # Create a new DataFrame to store the results
        results = pd.DataFrame(columns=[
            'Text', 'Flesch-Kincaid Score', 'New Dale-Chall', 'SMOG Index', 'Lexical Diversity', 'Top Words', 
            'Neg Sentiment', 'Neu Sentiment', 'Pos Sentiment', 'Compound Sentiment', 'Final Sentiment',
            'Top CTA Words', 'Sales-y Words Count', 'News-y Words Count'
        ])
        
        # Initialize total CTA counts
        total_cta_counts = {word: 0 for word in cta_words}
        
        # Process each row
        for index, row in df.iterrows():
            text_data = str(row[column])
            
            # Flesch-Kincaid score
            flesch_kincaid_score = textstat.flesch_kincaid_grade(text_data)
            
            # New Dale-Chall score
            dale_chall_score = new_dale_chall(text_data)
            
            # SMOG Index
            smog_score = smog_index(text_data)
            
            # Lexical diversity
            words = nltk.word_tokenize(text_data.lower())
            lexical_diversity = len(set(words)) / len(words) if words else 0
            
            # Top-performing words
            fdist = FreqDist(words)
            top_words = fdist.most_common(10)
            
            # Sentiment analysis
            sentiment = analyzer.polarity_scores(text_data)
            neg_sentiment = sentiment['neg']
            neu_sentiment = sentiment['neu']
            pos_sentiment = sentiment['pos']
            compound_sentiment = sentiment['compound']
            
            # Determine final sentiment
            if compound_sentiment > 0.05:
                final_sentiment = 'Positive'
            elif compound_sentiment < -0.05:
                final_sentiment = 'Negative'
            else:
                final_sentiment = 'Neutral'
            
            # Top-performing CTA words
            cta_word_counts = {word: words.count(word) for word in cta_words}
            
            # Update total CTA counts
            for word, count in cta_word_counts.items():
                total_cta_counts[word] += count
            
            # "Sales-y" vs "News-y" words
            salesy_count = sum(words.count(word) for word in salesy_words)
            newsy_count = sum(words.count(word) for word in newsy_words)
            
            # Append results to the DataFrame
            new_row = pd.DataFrame({
                'Text': [text_data],
                'Flesch-Kincaid Score': [flesch_kincaid_score],
                'New Dale-Chall': [dale_chall_score],
                'SMOG Index': [smog_score],
                'Lexical Diversity': [lexical_diversity],
                'Top Words': [str(top_words)],
                'Neg Sentiment': [neg_sentiment],
                'Neu Sentiment': [neu_sentiment],
                'Pos Sentiment': [pos_sentiment],
                'Compound Sentiment': [compound_sentiment],
                'Final Sentiment': [final_sentiment],
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
        st.write('**New Dale-Chall:** A readability formula that considers word difficulty and sentence length to assess text difficulty.')
        st.write('**SMOG Index:** A readability formula that estimates the years of education needed to understand a piece of writing.')
        st.write('**Lexical Diversity:** A measure of how many different words are used in the text. It is calculated as the ratio of unique words to the total number of words. A higher ratio indicates a more diverse vocabulary.')
        st.write('**Top Words:** The most frequently occurring words in the text. This helps identify common themes or topics.')
        st.write('**Sentiment Analysis:** An assessment of the emotional tone of the text. The sentiment score ranges from -1 (very negative) to 1 (very positive). A score close to 0 indicates neutral sentiment.')
        st.write('**Top CTA Words:** The most frequently occurring call-to-action words in the text. These words are often used to encourage readers to take specific actions.')
        st.write('**Sales-y Words Count:** The number of words in the text that are typically associated with sales language. A higher count indicates a more promotional tone.')
        st.write('**News-y Words Count:** The number of words in the text that are typically associated with news language. A higher count indicates a more informational tone.')
        
        # Display charts for better visualization
        st.write('### Charts:')
        st.bar_chart(results[['Flesch-Kincaid Score', 'New Dale-Chall', 'SMOG Index', 'Lexical Diversity', 'Sales-y Words Count', 'News-y Words Count']])
        
        # Additional visualizations
        st.write('### Sentiment Distribution:')
        sentiment_df = pd.DataFrame(results[['Neg Sentiment', 'Neu Sentiment', 'Pos Sentiment', 'Compound Sentiment']])
        st.bar_chart(sentiment_df)
        
        st.write('### Top Words Word Cloud:')
        # Debugging: Print total CTA counts
        st.write('Total CTA Counts:', total_cta_counts)
        
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(total_cta_counts)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)
