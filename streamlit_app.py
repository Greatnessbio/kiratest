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
nltk.download('stopwords')

# Load the vader_lexicon resource
sia = SentimentIntensityAnalyzer()

# Create a Streamlit app
st.title("Text Analysis App")

# Add a file uploader
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

# Define the function globally
def syllables_per_word(word, d):
    return [len(list(y for y in x if y[-1].isdigit())) for x in d[word.lower()]]

if uploaded_file is not None:
    try:
        # Read the CSV file
        df = pd.read_csv(uploaded_file)

        # Ask the user if they want to use the header row
        use_header = st.checkbox("Use header row")

        if use_header:
            text_input = df.iloc[:, 0].tolist()  # Convert to list
        else:
            text_input = df.iloc[:, 0].apply(lambda x: x.lower()).tolist()  # Convert to list

        # Add a button to analyze the text
        if st.button("Analyze Text"):
            # Data Cleaning
            stop_words = set(nltk.corpus.stopwords.words('english'))
            cleaned_texts = []
            for text in text_input:
                tokens = word_tokenize(text)
                filtered_tokens = [w for w in tokens if w not in stop_words and w.isalnum()]
                cleaned_texts.append(' '.join(filtered_tokens))

            # Tokenize the input text
            tokens = [word_tokenize(text) for text in cleaned_texts]

            # Calculate the Flesch-Kincaid score
            d = cmudict.dict()
            syllables = 0
            total_words = 0
            for token in tokens:
                for word in token:
                    try:
                        syllables += [len(list(y for y in x if y[-1].isdigit())) for x in d[word.lower()]][0]
                    except KeyError:
                        syllables += syllables_per_word(word, d)
                    total_words += 1

            # Calculate Flesch-Kincaid, handling potential division by zero
            if total_words > 0:
                flesch_kincaid_score = 0.39 * (total_words / len(text_input)) + 11.8 * (syllables / total_words) - 15.59
            else:
                flesch_kincaid_score = 0  # Or assign a suitable value when there are no words

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
            sentiment_scores = [sia.polarity_scores(text) for text in cleaned_texts]
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

    except Exception as e:
        st.error(f"An error occurred: {e}")
