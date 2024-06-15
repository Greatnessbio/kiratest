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
    try:
        return [len(list(y for y in x if y[-1].isdigit())) for x in d[word.lower()]][0]
    except KeyError:
        # If word is an acronym or not in CMU dict, count vowels
        return sum(1 for letter in word if letter.lower() in 'aeiouy')

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
                    syllables += syllables_per_word(word, d)
                    total_words += 1

            # Calculate Flesch-Kincaid, handling potential division by zero
            if total_words > 0:
                flesch_kincaid_score = 0.39 * (total_words / len(text_input)) + 11.8 * (syllables / total_words) - 15.59
            else:
                flesch_kincaid_score = 0  # Or assign a suitable value when there are no words

            st.write("Flesch-Kincaid Score:", flesch_kincaid_score)

            # ... (rest of your code - lexical diversity, sentiment analysis, etc.)

    except Exception as e:
        st.error(f"An error occurred: {e}")
