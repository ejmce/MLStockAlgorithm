from logging import raiseExceptions
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def scrape_google_news(ticker):
    url = f"https://news.google.com/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US%3Aen"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    articles_data = []
    articles = soup.find_all('article')
    
    # Get the current date and three days ago
    today = datetime.now()
    three_days_ago = today - timedelta(days=3)
    
    for article in articles:
        title_tag = article.find('a', {'class': 'JtKRv'})
        if title_tag:
            title = title_tag.text.strip()
        else:
            continue

        link = article.find('a', {'class': 'JtKRv'} )['href']
        full_link = f"https://news.google.com{link[1:]}"  # Complete the relative URL
            
        # Extract and parse the published date
        published_time_tag = article.find('time')
        if published_time_tag:
            published_time = datetime.strptime(published_time_tag['datetime'], '%Y-%m-%dT%H:%M:%SZ')
        else:
            continue
            
        # Filter for articles from the last 3 days
        if published_time >= three_days_ago:
            articles_data.append({
                'Ticker': ticker,
                'Title': title,
                'Published At': published_time.strftime('%Y-%m-%d %H:%M:%S'),
                'URL': full_link,
            })
    
    return articles_data

def get_recent_news(tickers):
    # Initialize an empty DataFrame to store the sentiment data for all tickers
    average_sentiment_df = pd.DataFrame()
    
    for ticker in tickers:
        try:
            recent_news = scrape_google_news(ticker)
    
            # Perform sentiment analysis
            recent_news_df = pd.DataFrame(recent_news)
            sentiment_analyzed_df = perform_sentiment_analysis_on_news(recent_news_df)
            # Calculate the average sentiment for each ticker
            stock_average_sentiment_df = calculate_average_sentiment(sentiment_analyzed_df)
            # Append the result to the cumulative DataFrame
        except Exception as e:
            # If an error occurs, create a DataFrame with the ticker and a sentiment of 0
            stock_average_sentiment_df = pd.DataFrame({'Ticker': [ticker], 'Average Sentiment': [0]})

        average_sentiment_df = pd.concat([average_sentiment_df, stock_average_sentiment_df], ignore_index=True)
    
    return average_sentiment_df

def analyze_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_score = analyzer.polarity_scores(text)
    return sentiment_score

def perform_sentiment_analysis_on_news(df):
    df['Sentiment'] = df['Title'].apply(lambda x: analyze_sentiment(x)['compound'])
    return df

def calculate_average_sentiment(df):
    average_sentiment_df = df.groupby('Ticker')['Sentiment'].mean().reset_index()
    average_sentiment_df.rename(columns={'Sentiment': 'Average Sentiment'}, inplace=True)
    return average_sentiment_df

# Example usage
# news_articles_df = get_news_articles(tickers)
# print(news_articles_df.head())
