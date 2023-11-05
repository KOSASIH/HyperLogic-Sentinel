import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter

def analyze_textual_data(text_data):
    # Tokenize the text data
    tokens = word_tokenize(text_data)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
    
    # Perform sentiment analysis
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = sid.polarity_scores(text_data)
    
    # Identify keywords using word frequency
    word_freq = Counter(filtered_tokens)
    top_keywords = word_freq.most_common(10)
    
    # Generate markdown report
    report = f"### Textual Data Analysis Report\n\n"
    report += f"**Sentiment Analysis:**\n\n"
    report += f"Positive Sentiment: {sentiment_scores['pos']:.2f}\n"
    report += f"Negative Sentiment: {sentiment_scores['neg']:.2f}\n"
    report += f"Neutral Sentiment: {sentiment_scores['neu']:.2f}\n"
    report += f"Compound Sentiment: {sentiment_scores['compound']:.2f}\n\n"
    report += f"**Top Keywords:**\n\n"
    for keyword, frequency in top_keywords:
        report += f"- {keyword}: {frequency}\n"
    
    return report

# Example usage
text_data = "This is a sample text. It talks about security threats and suspicious activities. " \
            "We need to analyze it and generate a report."
report = analyze_textual_data(text_data)
print(report)
