from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import re
from bs4 import BeautifulSoup

# Get dataroma top "big bets" tickers
def get_top_bb_tickers():
    # Setup Selenium with Chrome WebDriver
    options = Options()
    options.headless = True  # Run Chrome in headless mode (without a GUI)
    service = Service(ChromeDriverManager().install())
    
    driver = webdriver.Chrome(service=service, options=options)
    
    # Navigate to the page
    driver.get('https://www.dataroma.com/m/g/portfolio.php?pct=0&o=p')

    # Wait for the page to load completely
    driver.implicitly_wait(10)
    
    # Get page source after JavaScript execution
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    
    # Close the driver
    driver.quit()
    
    # Scraping tickers from the table
    tickers = []
    # Define a regular expression pattern to match the ticker symbol
    pattern = re.compile(r'^[A-Z]+')
    # Find all <td> elements with class 'stock'
    td_elements = soup.find_all('td', {'class': 'sym'})
    
    # Extract ticker symbols
    for td in td_elements:
        a_tag = td.find('a')
        if a_tag:
            # Extract the text and match the ticker symbol
            text = a_tag.get_text(strip=True)
            match = pattern.match(text)
            if match:
                ticker = match.group(0)
                tickers.append(ticker)
    
    return tickers

def get_brk_tickers():
    # Setup Selenium with Chrome WebDriver
    options = Options()
    options.headless = True  # Run Chrome in headless mode (without a GUI)
    service = Service(ChromeDriverManager().install())
    
    driver = webdriver.Chrome(service=service, options=options)
    
    # Navigate to the page
    driver.get('https://www.dataroma.com/m/holdings.php?m=brk')

    # Wait for the page to load completely
    driver.implicitly_wait(10)
    
    # Get page source after JavaScript execution
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    
    # Close the driver
    driver.quit()
    
    # Scraping tickers from the table
    tickers = []
    # Define a regular expression pattern to match the ticker symbol
    pattern = re.compile(r'^[A-Z]+')
    # Find all <td> elements with class 'stock'
    td_elements = soup.find_all('td', {'class': 'stock'})
    
    # Extract ticker symbols
    for td in td_elements:
        a_tag = td.find('a')
        if a_tag:
            # Extract the text and match the ticker symbol
            text = a_tag.get_text(strip=True)
            match = pattern.match(text)
            if match:
                ticker = match.group(0)
                tickers.append(ticker)
    
    return tickers

# Combine the tickers from Top Big Bets and BRK
def get_combined_tickers():
    top_tickers = get_top_bb_tickers()
    brk_tickers = get_brk_tickers()
    combined_tickers = list(set(top_tickers + brk_tickers))
    df = pd.DataFrame(combined_tickers, columns=['Ticker'])
    return df

# Example usage
# tickers = get_combined_tickers()
# print(tickers)
