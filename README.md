Stock trading algorithm using ML to analyze historical data.

Start with ML Stock Algo High Overview and ML Stock Algo Planning. Follow through ML Stock Algo Planning stages.


Uploaded v1 08/14/2024 - Issues: BeautifulSoup and Selinium struggled obtaining the tickers for congress trades. Scrapping that plan and instead getting the tickers from BRK holdings, and Top "big bets" from Dataroma. News API only allows for 50 API calls within 12 hours. Need to find a different method for obtaining potential catalysts.

Updated v1.01 08/15/2024 Switched from News API to scraping google news. Performed a sentiment analysis on the news, retrieved the avg. Created a new df with Ticker, Price, Sentiment.

Updated v1.02 08/20/2024 Taking the data collected, ran LSTM and Linear Regression test to create the model.

Updated v1.02 08/28/2024 Created API endpoints to execute different functions, such as training, getting a singular ticker, and retrieving a list of tickers to buy/sell
