import bs4 as bs
import requests
import yfinance as yf
import datetime

start = datetime.datetime(2018, 1, 1)
end = datetime.datetime(2024, 1, 1)

spy = yf.download("SPY", start=start, end=end)
if spy is not None:
    print(spy)
    spy.to_csv("spy_daily.csv")
else:
    print("Failed to download SPY data.")


resp = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
soup = bs.BeautifulSoup(resp.text, "lxml")
table = soup.find("table", {"class": "wikitable sortable"})

tickers = []

if isinstance(table, bs.Tag):
    for row in table.find_all("tr")[1:]:
        if isinstance(row, bs.Tag):
            ticker = row.find_all("td")[0].text
        tickers.append(ticker)
else:
    print("Failed to find the S&P 500 table on the Wikipedia page or the table is not a valid Tag object.")

print(tickers)

tickers = [s.replace("\n", "") for s in tickers]


data = yf.download(tickers, start=start, end=end)

if data is not None:
    df = (
        data.stack()
        .reset_index()
        .rename(index=str, columns={"level_1": "Symbol"})
        .sort_values(["Symbol", "Date"])
    )
    df.set_index("Date", inplace=True)

    df.to_csv("sp500_stocks.csv")
else:
    print("Failed to download data for the tickers.")
