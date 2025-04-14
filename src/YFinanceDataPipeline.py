import bs4 as bs
import requests
import yfinance as yf
import datetime

start = datetime.datetime(2018, 1, 1)
end = datetime.datetime(2024, 1, 1)

spy = yf.download("SPY", start=start, end=end)
print(spy)
spy.to_csv("spy_daily.csv")


resp = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
soup = bs.BeautifulSoup(resp.text, "lxml")
table = soup.find("table", {"class": "wikitable sortable"})


tickers = []

for row in table.findAll("tr")[1:]:
    ticker = row.findAll("td")[0].text
    tickers.append(ticker)

print(tickers)

tickers = [s.replace("\n", "") for s in tickers]


data = yf.download(tickers, start=start, end=end)


df = (
    data.stack()
    .reset_index()
    .rename(index=str, columns={"level_1": "Symbol"})
    .sort_values(["Symbol", "Date"])
)
df.set_index("Date", inplace=True)

df.to_csv("sp500_stocks.csv")
