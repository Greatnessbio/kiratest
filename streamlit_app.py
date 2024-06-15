import nltk
import streamlit as st
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import cmudict
from nltk.probability import FreqDist
from collections import Counter
import re
import csv
import matplotlib.pyplot as plt

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
        # Ask the user if the CSV has a header
        has_header = st.checkbox("CSV file has header row")

        # Read the CSV file
        if has_header:
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file, header=None)

        # Display CSV preview
        st.subheader("CSV Data Preview:")
        st.write(df.head())

        # Get column names or indices for dropdown
        if has_header:
            column_options = list(df.columns)
        else:
            column_options = [str(i) for i in range(len(df.columns))]

        # Dropdown for column selection
        selected_column = st.selectbox("Select the text column:", column_options)

        # Use the selected column for text input
        if has_header:
            text_input = df[selected_column].tolist()
        else:
            text_input = df.iloc[:, int(selected_column)].tolist()

        # --- User Input for "Sales-y" and "News-y" Words ---
        st.subheader("Customize Word Categories:")
        sales_y_words_input = st.text_input("Enter 'Sales-y' words (comma-separated):", "buy,sale,discount,offer,deal,free,trial,demo")
        news_y_words_input = st.text_input("Enter 'News-y' words (comma-separated):", "news,update,article,blog,post,story,report")

        # Convert input strings to lists
        sales_y_words = [word.strip().lower() for word in sales_y_words_input.split(",")]
        news_y_words = [word.strip().lower() for word in news_y_words_input.split(",")]

        # Add a button to analyze the text
        if st.button("Analyze Text"):
            # --- Analysis Results Calculation (ALL CALCULATIONS INSIDE THE LOOP) ---
            results = []
            for text in text_input:
                try:
                    # Data Cleaning and Tokenization for the current text
                    stop_words = set(nltk.corpus.stopwords.words('english'))
                    tokens = word_tokenize(text)
                    filtered_tokens = [w for w in tokens if w.lower() not in stop_words and (w.isalnum() or '-' in w or "'" in w)]
                    cleaned_text = ' '.join(filtered_tokens)

                    # Calculate analysis for the current text
                    d = cmudict.dict()
                    syllables = 0
                    total_words = 0  # Reset total_words for each text

                    for word in filtered_tokens:
                        syllables += syllables_per_word(word, d)
                        total_words += 1

                    # Calculate Flesch-Kincaid Score
                    total_sentences = len(sent_tokenize(cleaned_text))
                    if total_words > 0:
                        flesch_kincaid_score = 0.39 * (total_words / total_sentences) + 11.8 * (syllables / total_words) - 15.59
                    else:
                        flesch_kincaid_score = 0

                    # Calculate lexical diversity
                    lexical_diversity = len(set(filtered_tokens)) / len(filtered_tokens) if filtered_tokens else 0

                    # Perform sentiment analysis on cleaned text
                    sentiment_score = sia.polarity_scores(cleaned_text)
                    sentiment = "Positive" if sentiment_score['compound'] > 0.05 else "Negative" if sentiment_score['compound'] < -0.05 else "Neutral"

                    # Calculate "sales-y" vs "news-y" words
                    sales_y_count = sum(1 for word in filtered_tokens if word.lower() in sales_y_words)
                    news_y_count = sum(1 for word in filtered_tokens if word.lower() in news_y_words)
                    sales_y_score = sales_y_count / total_words if total_words > 0 else 0
                    news_y_score = news_y_count / total_words if total_words > 0 else 0

                    # Store results in a dictionary
                    result = {
                        "Original Text": text,
                        "Flesch-Kincaid Score": flesch_kincaid_score,
                        "Lexical Diversity": lexical_diversity,
                        "Sentiment": sentiment,
                        "Sentiment Score (Compound)": sentiment_score['compound'],
                        "Sales-y Score": sales_y_score,
                        "News-y Score": news_y_score
                    }
                    results.append(result)
                except ZeroDivisionError as e:
                    st.error(f"Error calculating scores for text: {text}. {e}")
                    continue  # Skip to the next text

            # Create a DataFrame from the results list
            results_df = pd.DataFrame(results)

            # --- Output Section ---
            st.subheader("Analysis Results:")
            st.write(results_df)

            # --- Explanations ---
            st.subheader("Explanations:")
            st.write("**Flesch-Kincaid Score:** Measures readability (higher = easier).")
            st.write("**Lexical Diversity:** Ratio of unique words to total words (higher = more varied vocabulary).")
            st.write("**Sentiment:** Overall sentiment (positive, negative, neutral).")
            st.write("**Sentiment Score (Compound):** A normalized score from -1 (most negative) to 1 (most positive).")
            st.write("**Sales-y Score:** Proportion of words related to sales and marketing.")
            st.write("**News-y Score:** Proportion of words related to news and information.")

            # --- Additional Insights ---
            st.subheader("Additional Insights:")
            top_words = Counter([word for sublist in tokens for word in sublist]).most_common(10)
            st.write("**Top Performing Words:**", top_words)

            # Bar chart for top words
            fig, ax = plt.subplots()
            ax.bar([word[0] for word in top_words], [word[1] for word in top_words])
            ax.set_xlabel("Words")
            ax.set_ylabel("Frequency")
            ax.set_title("Top 10 Words")
            st.pyplot(fig)

            cta_words = [word for sublist in tokens for word in sublist if word.lower() in ["buy", "sign", "register", "learn", "download", "get", "start", "try", "join", "explore"]]
            top_cta_words = Counter(cta_words).most_common(10)
            st.write("**Top Performing CTA Words:**", top_cta_words)

            # Bar chart for top CTA words
            fig, ax = plt.subplots()
            ax.bar([word[0] for word in top_cta_words], [word[1] for word in top_cta_words])
            ax.set_xlabel("CTA Words")
            ax.set_ylabel("Frequency")
            ax.set_title("Top 10 CTA Words")
            st.pyplot(fig)

    except (FileNotFoundError, pd.errors.EmptyDataError) as e:
        st.error(f"Error reading CSV file: {e}")
    except KeyError as e:
        st.error(f"Error selecting column: {e}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
