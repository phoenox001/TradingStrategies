# import bs4 as bs
# import requests
# import yfinance as yf
# import datetime

# start = datetime.datetime(2018, 1, 1)
# end = datetime.datetime(2024, 1, 1)

# spy = yf.download("SPY", start=start, end=end)
# if spy is not None:
#     print(spy)
#     spy.to_csv("./Data/spy_daily.csv")
# else:
#     print("Failed to download SPY data.")


# resp = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
# soup = bs.BeautifulSoup(resp.text, "lxml")
# table = soup.find("table", {"class": "wikitable sortable"})

# tickers = []

# if isinstance(table, bs.Tag):
#     for row in table.find_all("tr")[1:]:
#         if isinstance(row, bs.Tag):
#             ticker = row.find_all("td")[0].text
#         tickers.append(ticker)
# else:
#     print(
#         "Failed to find the S&P 500 table on the Wikipedia page or the table is not a valid Tag object."
#     )

# print(tickers)

# tickers = [s.replace("\n", "") for s in tickers]


# data = yf.download(tickers, start=start, end=end)

# if data is not None:
#     df = (
#         data.stack()
#         .reset_index()
#         .rename(index=str, columns={"level_1": "Symbol"})
#         .sort_values(["Symbol", "Date"])
#     )
#     df.set_index("Date", inplace=True)

#     df.to_csv("../Data/sp500_stocks.csv")
# else:
#     print("Failed to download data for the tickers.")


import pandas as pd
import yfinance as yf
import logging
import ta

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# check if file is already available
# try:
#     data = pd.read_csv("../data/stock_data.csv")
#     data["Date"] = pd.to_datetime(data["Date"])
#     data.set_index("Date", inplace=True)
#     data.sort_values(by=["ticker", "Date"], inplace=True)
#     logging.info("Data already available, skipping download.")
# except FileNotFoundError:

# Read ticker data from CSV
filepath = (
    "/Users/tim/Documents/Projects/TradingStrategies/data/ticker_Symbols_yfinance.csv"
)
tickers = pd.read_csv(filepath)
tickers = tickers["ticker"].tolist()
tickers = list(set(tickers))
logging.info(f"Number of tickers: {len(tickers)}")
data = {}

# Download stock price data for each ticker
for ticker in tickers:
    df = pd.DataFrame()
    df = yf.download(ticker, start="2014-01-01", end="2024-01-01")
    # check if download was successful
    if df is None or df.empty:
        continue
    df["Ticker"] = ticker
    data[ticker] = df
    logging.info(f"Downloaded data for {ticker}")
    logging.info(f"Number of tickers with data: {len(data)}")

logging.info("Data download complete.")

# Concatenate all dataframes into a single dataframe
stock_data = pd.concat(data.values(), axis=0)
stock_data.reset_index(inplace=True)
stock_data["Date"] = pd.to_datetime(stock_data["Date"])
stock_data.sort_values(by=["Ticker", "Date"], inplace=True)
logging.info("Data concatenation complete.")

# Save the data to a CSV file
stock_data.to_csv("../data/stock_data.csv", index=False)
print(stock_data.head())
logging.info("Data saved to CSV file.")
