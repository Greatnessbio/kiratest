import nltk
import streamlit as st
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import cmudict
from nltk.probability import FreqDist
from collections import Counter
import re
import csv

# Specify the path to the nltk_data directory
nltk.data.path.append('./nltk_data')

# Download the required NLTK resources
nltk.download('vader_lexicon')
nltk.download('cmudict')
nltk.download('punkt')

# Load the vader_lexicon resource
sia = SentimentIntensityAnalyzer()

# Create a Streamlit app
st.title("Text Analysis App")

# Add a file uploader
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the CSV file
    df = pd.read_csv(uploaded_file)

    # Ask the user if they want to use the header row
    use_header = st.checkbox("Use header row")

    if use_header:
        text_input = df.iloc[:, 0]
    else:
        text_input = df.iloc[:, 0].apply(lambda x: x.lower())

    # Add a button to analyze the text
    if st.button("Analyze Text"):
        # Tokenize the input text
        tokens = [word_tokenize(text) for text in text_input]

        # Calculate the Flesch-Kincaid score
        d = cmudict.dict()
        syllables = 0
        for token in tokens:
            for word in token:
                try:
                    syllables += [len(list(y for y in x if y[-1].isdigit())) for x in d[word.lower()]][0]
                except KeyError:
                    syllables += syllables_per_word(word, d)
        flesch_kincaid_score = 0.39 * (len(tokens) / len(re.split('[.!?]', ' '.join(text_input)))) + 0.11 * (syllables / len([word for token in tokens for word in token]))
        st.write("Flesch-Kincaid Score:", flesch_kincaid_score)

        # Calculate lexical diversity
        lexical_diversity = len(set([word for token in tokens for word in token])) / len([word for token in tokens for word in token])
        st.write("Lexical Diversity:", lexical_diversity)

        # Calculate top-performing words
        top_words = Counter([word for token in tokens for word in token]).most_common(10)
        st.write("Top-performing Words:")
        st.write(top_words)

        # Calculate top-performing CTA words
        cta_words = [word for token in tokens for word in token if word.lower() in ["buy", "sign", "register", "learn", "download", "get", "start", "try", "join", "explore"]]
        top_cta_words = Counter(cta_words).most_common(10)
        st.write("Top-performing CTA Words:")
        st.write(top_cta_words)

        # Perform sentiment analysis
        sentiment_scores = [sia.polarity_scores(text) for text in text_input]
        st.write("Sentiment Scores:")
        st.write(sentiment_scores)

        # Determine the sentiment (positive, negative, or neutral)
        compound_scores = [score['compound'] for score in sentiment_scores]
        sentiments = []
        for score in compound_scores:
            if score > 0.05:
                sentiments.append("Positive")
            elif score < -0.05:
                sentiments.append("Negative")
            else:
                sentiments.append("Neutral")
        st.write("Sentiments:")
        st.write(sentiments)

        # Calculate "sales-y" vs "news-y" words
        sales_y_words = [word for token in tokens for word in token if word.lower() in ["buy", "sale", "discount", "offer", "deal", "free", "trial", "demo"]]
        news_y_words = [word for token in tokens for word in token if word.lower() in ["news", "update", "article", "blog", "post", "story", "report"]]
        sales_y_score = len(sales_y_words) / len([word for token in tokens for word in token])
        news_y_score = len(news_y_words) / len([word for token in tokens for word in token])
        st.write("Sales-y Score:", sales_y_score)
        st.write("News-y Score:", news_y_score)

def syllables_per_word(word, d):
    return [len(list(y for y in x if y[-1].isdigit())) for x in d[word.lower()]]
