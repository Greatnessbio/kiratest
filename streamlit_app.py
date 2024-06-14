import pandas as pd
from textstat import textstat
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import nltk

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('vader_lexicon')
nltk.download('stopwords')

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
    # Prompt user for input
    file_path = input("Enter the path to the CSV file: ")
    text_column = input("Enter the name of the column containing text data: ")
    
    cta_words = input("Enter CTA words separated by commas: ").split(',')
    salesy_words = input("Enter sales-y words separated by commas: ").split(',')
    newsy_words = input("Enter news-y words separated by commas: ").split(',')
    custom_stopwords = input("Enter additional stop words separated by commas: ").split(',')
    
    # Combine custom stopwords with default NLTK stopwords
    custom_stopwords = set(custom_stopwords) | set(stopwords.words('english'))
    
    # Load the data
    df = pd.read_csv(file_path)
    
    # Initialize sentiment analyzer
    sia = SentimentIntensityAnalyzer()
    
    # Apply the analysis to each row
    df['Analysis'] = df[text_column].apply(lambda text: analyze_text(text, cta_words, salesy_words, newsy_words, custom_stopwords, sia))
    
    # Expand the analysis dictionary into separate columns
    analysis_df = df['Analysis'].apply(pd.Series)
    result_df = pd.concat([df, analysis_df], axis=1).drop(columns=['Analysis'])
    
    # Save the result to a new CSV file
    result_df.to_csv('analyzed_data.csv', index=False)
    print("Analysis complete. Results saved to 'analyzed_data.csv'.")

# Run the main function
if __name__ == "__main__":
    main()
