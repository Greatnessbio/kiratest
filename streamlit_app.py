import nltk
import streamlit as st
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import cmudict
from nltk.probability import FreqDist
from collections import Counter
import re

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

# Add a text input field
text_input = st.text_input("Enter some text:")

# Add a button to analyze the text
if st.button("Analyze Text"):
    # Tokenize the input text
    tokens = word_tokenize(text_input)

    # Calculate the Flesch-Kincaid score
    syllables = 0
    d = cmudict.dict()
    for word in tokens:
        try:
            syllables += [len(list(y for y in x if y[-1].isdigit())) for x in d[word.lower()]][0]
        except KeyError:
            syllables += syllables_per_word(word)
    flesch_kincaid_score = 0.39 * (len(tokens) / len(re.split('[.!?]', text_input))) + 0.11 * (syllables / len(tokens))
    st.write("Flesch-Kincaid Score:", flesch_kincaid_score)

    # Calculate lexical diversity
    lexical_diversity = len(set(tokens)) / len(tokens)
    st.write("Lexical Diversity:", lexical_diversity)

    # Calculate top-performing words
    top_words = Counter(tokens).most_common(10)
    st.write("Top-performing Words:")
    st.write(top_words)

    # Calculate top-performing CTA words
    cta_words = [word for word in tokens if word.lower() in ["buy", "sign", "register", "learn", "download", "get", "start", "try", "join", "explore"]]
    top_cta_words = Counter(cta_words).most_common(10)
    st.write("Top-performing CTA Words:")
    st.write(top_cta_words)

    # Perform sentiment analysis
    sentiment_scores = sia.polarity_scores(text_input)
    st.write("Sentiment Scores:")
    st.write(sentiment_scores)

    # Determine the sentiment (positive, negative, or neutral)
    compound_score = sentiment_scores['compound']
    if compound_score > 0.05:
        st.write("Sentiment: Positive")
    elif compound_score < -0.05:
        st.write("Sentiment: Negative")
    else:
        st.write("Sentiment: Neutral")

    # Calculate "sales-y" vs "news-y" words
    sales_y_words = [word for word in tokens if word.lower() in ["buy", "sale", "discount", "offer", "deal", "free", "trial", "demo"]]
    news_y_words = [word for word in tokens if word.lower() in ["news", "update", "article", "blog", "post", "story", "report"]]
    sales_y_score = len(sales_y_words) / len(tokens)
    news_y_score = len(news_y_words) / len(tokens)
    st.write("Sales-y Score:", sales_y_score)
    st.write("News-y Score:", news_y_score)

def syllables_per_word(word):
    return [len(list(y for y in x if y[-1].isdigit())) for x in d[word.lower()]]

d = cmudict.dict()
