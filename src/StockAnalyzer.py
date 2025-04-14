import pandas as pd
import numpy as np
import pandas_datareader.data as web
import datetime as dt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

"""
TODO:
- show description of the selected indicators
- download bitcoin and ethereum data
"""


@st.cache_data
def initialize():
    sp500_stock_data = pd.read_csv(
        "/Users/tim/Documents/Projects/PythonProjects/Trading/Data/YahooFinanceHistoricalPriceData/sp500_stocks.csv"
    )
    sp500_stock_data["Date"] = sp500_stock_data["Date"].astype("datetime64[ns]")
    sp500_stock_names = pd.read_csv(
        "/Users/tim/Documents/Projects/PythonProjects/Trading/Data/YahooFinanceHistoricalPriceData/sp500_companies.csv"
    )
    sp500_stock_names.loc[len(sp500_stock_names.index)] = [
        "SPY",
        "SPY",
        "Index",
    ]
    sp500_stock_names = sp500_stock_names.set_index("Symbol")

    spy_data = pd.read_csv(
        "/Users/tim/Documents/Projects/PythonProjects/Trading/Data/YahooFinanceHistoricalPriceData/spy_daily.csv"
    )
    spy_data["Date"] = spy_data["Date"].astype("datetime64[ns]")
    spy_data["Open"] = spy_data["Open"].astype("float")
    spy_data["High"] = spy_data["High"].astype("float")
    spy_data["Low"] = spy_data["Low"].astype("float")
    spy_data["Close"] = spy_data["Close"].astype("float")
    spy_data["Volume"] = spy_data["Volume"].astype("float")
    spy_data["Symbol"] = "SPY"

    # combine all three dataframes
    sp500_stock_data = pd.concat([sp500_stock_data, spy_data])
    sp500_stock_data = sp500_stock_data.reset_index()
    sp500_stock_data = sp500_stock_data.join(sp500_stock_names, on="Symbol", how="left")
    st.write(sp500_stock_data)
    # reindex with index as stock symbols
    sp500_stock_data = sp500_stock_data.set_index(["Symbol", "Sector", "Date"])

    indicators = [
        "SMA",
        "EWMA",
        "Accumulation Distribution",
        "ADX",
        "Aroon Oscillator",
        "ATR Bands",
        "ATR Trailing Stops",
        "Average True Range",
        "Bollinger Bands",
        "Bollinger Band Width",
        "Bollinger %%b",
        "Candlestick Patterns",
        "Chaikin Money Flow",
        "Chaikin Oscillator",
        "Chaikin Volatility",
        "Chande Momentum Oscillator",
        "Chandelier Exits",
        "Choppiness Index",
        "Compare Prices",
        "Coppock Indicator",
        "Detrended Price Oscillator",
        "Directional Movement Index",
        "Displaced Moving Average",
        "Donchian Channels",
        "Ease of Movement",
        "Elder Ray Index",
        "Equivolume Charts",
        "Fibonacci Extensions",
        "Fibonacci Retracements",
        "Force Index",
        "Heikin-Ashi Candlesticks",
        "Hull Moving Average",
        "Ichimoku Cloud",
        "Iverted Axis",
        "Keltner Channels",
        "KST Indicator",
        "Linear Regression",
        "Linear Regression Indicator",
        "MA Oscillator",
        "MACD Indicator",
        "MACD Histogramm",
        "MACD Percentage",
        "Mass Index",
        "Median Price",
        "Momentum Indicator",
        "Money Flow Index",
        "Moving Average Filters",
        "Moving Average High/Low/Open",
        "Multiple Moving Averages",
        "Negative Volume",
        "On Balance Volume",
        "Parablic SAR",
        "Percentage Bands",
        "Percentage Trailing Stops",
        "Pivot Points",
        "Positive Volume",
        "Price Comparison",
        "Price Differential",
        "Price Envelope",
        "Price Ratio",
        "Price Volume Trend",
        "Rainbow 3D Moving Averages",
        "Rate of Change (Price)",
        "Rate of Change (Volume)",
        "Relative Strength (Compare)",
        "Relative Strength Index (RSI)",
        "Safezone Indicator",
        "Slow Stochastic",
        "Smoothed Rate of Change (SROC)",
        "Standard Deviation Channels",
        "Stochastic Oscillator",
        "Stochastic RSI",
        "Trend Lines",
        "TRIX Indicator",
        "True Range",
        "Twiggs Momentum Oscillator",
        "Twiggs Money Flow",
        "Twiggs Proprietary Indicators",
        "Twiggs Smoothed Momentum",
        "Twiggs Trend Index",
        "Twiggs Volatility",
        "Typical Price",
        "Ultimate Oscillator",
        "Vertical Horizontal Filter (VHF)",
        "Volatility",
        "Volatility Ratio",
        "Volatility Stops",
        "Volume",
        "Volume Oscillator",
        "Weighted Close",
        "Wilder Moving Average",
        "Williams %R",
        "Williams Accumulate Distribute",
        "Williams Accumulation Distribution",
    ]

    return sp500_stock_data, sp500_stock_names, indicators


# calculate different indicators
# Simple Moving Average
def sma(data, period):
    # set dataframe
    sma = pd.DataFrame()

    # calculate sma
    sma["SMA"] = data["Close"].rolling(period).mean()
    sma = sma.fillna(0)
    return sma


# Exponential Moving Average
def ema(data, period):
    # set dataframe
    ema = pd.DataFrame()

    # calculate ema
    ema["EWMA"] = data["Close"].ewm(alpha=1 / period, adjust=False).mean()
    ema = ema.fillna(0)
    return ema


# Moving Average Convergence Divergence
def macd(data):
    # set dataframe
    macd = pd.DataFrame()

    # calculate macd
    macd["MACD"] = (data["Close"].ewm(span=12, adjust=False).mean()) - (
        data["Close"].ewm(span=26, adjust=False).mean()
    )
    macd["Signal_Line"] = macd["MACD"].ewm(span=9, adjust=False).mean()
    return macd


# Relative Strength Index
def rsi(data):
    # calculate ups and downs and split them in two groups
    change = data["Close"].diff()
    change.fillna(0)

    change_up = change.copy()
    change_down = change.copy()

    change_up[change_up < 0] = 0
    change_down[change_down > 0] = 0

    # verify
    change.equals(change_up + change_down)

    # set dataframe
    rsi = pd.DataFrame()

    # calculate moving average of ups and downs
    avg_up = change_up.rolling(14).mean()
    avg_down = change_down.rolling(14).mean().abs()

    # calculate rsi
    rsi = rsi.set_index("Date")
    rsi["RSI"] = 100 - (100 / (1 + (avg_up + avg_down)))
    return rsi


# Stochastic Oscillator
def stochOsc(data):
    # calculate highs and lows in 14 day period
    high = data["High"].rolling(14).max()
    high = high.fillna(0)
    low = data["Low"].rolling(14).min()
    low = low.fillna(0)

    # set dataframe
    stoch = pd.DataFrame()

    # calculate oscillator (fast)
    stoch["%K"] = ((data["Close"] - low) / (high - low)) * 100
    # calculate oscillator (slow)
    stoch["%D"] = stoch["%K"].rolling(3).mean()
    stoch = stoch.replace([np.inf, -np.inf], np.nan)
    stoch = stoch.fillna(0)
    return stoch


# true range
def trueRange(data):
    # prepare data
    tr = data.copy()
    high = data["High"]
    low = data["Low"]
    close = data["Close"]
    tr["tr0"] = abs(high - low)
    tr["tr1"] = abs(high - close.shift())
    tr["tr2"] = abs(low - close.shift())

    # calculate true_range
    true_range = pd.DataFrame()
    true_range["TR"] = tr[["tr0", "tr1", "tr2"]].max(axis=1)
    true_range = true_range.fillna(0)
    return true_range


# directional movement indicator
def directionalMovement(data, tr):
    move = pd.DataFrame()

    # calculate directional movement
    up = data["High"].diff()
    down = data["Low"].diff()
    smoothedUp = up.ewm(alpha=1 / 14).mean()
    smoothedDown = down.ewm(alpha=1 / 14).mean()

    move["Up_DI"] = 100 * (smoothedUp / tr["ATR"])
    move["Down_DI"] = abs(100 * (smoothedDown.values / tr["ATR"]))

    return move


# Average True Range
def atr(data):
    # set dataframe
    atr = pd.DataFrame()

    # calculate avg true range
    true_range = trueRange(data=data)
    atr["ATR"] = true_range.ewm(alpha=1 / 14, adjust=False).mean()
    return atr


# Average Directional Index
def adx(data):
    adx = pd.DataFrame()

    # get avg true range and directional movement
    tr = atr(data)
    dm = directionalMovement(data=data, tr=tr)

    # calculate adx and its variants
    adx["DX"] = (
        abs(dm["Up_DI"] - dm["Down_DI"]) / abs(dm["Up_DI"] + dm["Down_DI"])
    ) * 100
    adx["ADX"] = ((adx["DX"].shift(1) * (14 - 1)) + adx["DX"]) / 14
    adx["ADX_smooth"] = adx["ADX"].ewm(alpha=1 / 14).mean()
    adx = adx.fillna(0)
    return adx


# Bollinger Bands
def bollinger(data):
    bollinger = pd.DataFrame()
    # middle band
    bollinger["Mid"] = data["Close"].rolling(20).mean()

    # standard deviation
    std = data["Close"].rolling(window=20).std()

    # upper and lower bands
    bollinger["Lower"] = bollinger["Mid"] - 2 * std
    bollinger["Upper"] = bollinger["Mid"] + 2 * std
    return bollinger


# On-Balance Volume
def obv(data):
    obv = pd.DataFrame()
    # calculate obv
    change = data["Close"].diff()
    obv["OBV"] = (np.sign(change) * data["Volume"]).fillna(0).cumsum()
    return obv


def moneyFlowVolumeSeries(data):
    # calculate money flow
    mfv = pd.DataFrame()
    mfv["MFV"] = (
        data["Volume"]
        * (2 * data["Close"] - data["High"] - data["Low"])
        / (data["High"] - data["Low"])
    )
    return mfv


def moneyFlowVolume(data, n):
    # calculate money flow volume
    mfv = pd.DataFrame()
    mfv["MFV"] = moneyFlowVolumeSeries(data).rolling(n).sum()
    return mfv


# Chaikin Money Flow
def chaikinMoneyFlow(data):
    # calculate chaikin money flow
    cmf = pd.DataFrame()
    cmf["CMF"] = moneyFlowVolume(data, 20)["MFV"] / data["Volume"].rolling(20).sum()
    return cmf


# make basic chart with stock data
def makeChart(data):
    graph = go.Figure()
    graph.add_traces(
        go.Scatter(
            y=data.iloc[:, 0],
            x=data.index,
            mode="lines",
        )
    )
    return graph


# general function to calculate only the required indicators
def calculateIndicator(data, indicator_names):
    indicators = []
    i = 1

    # create chart figure
    stock_graph = make_subplots(
        rows=len(indicator_names) + 1,
        cols=1,
        shared_xaxes=True,
    )

    stock_graph.add_scatter(
        y=data["Close"],
        mode="lines",
        row=1,
        col=1,
        name=data["Name"].iloc[0],
    )

    # calculate selected indicators
    for indicator in indicator_names:
        # match indicators and calculate
        match indicator:
            case "SMA":
                period = st.number_input("What is the period of the SMA?", min_value=1)
                ind = sma(data, period)
                indicators.append(ind)

                count = st.checkbox("Multiple SMAs?")

                # create traces for chart
                stock_graph.add_scatter(
                    y=data["Close"],
                    mode="lines",
                    row=i + 1,
                    col=1,
                    name=data["Name"].iloc[0],
                )
                stock_graph.add_scatter(y=ind["SMA"], row=i + 1, col=1, name=indicator)

                # check if multiple graphs are necessary
                if count:
                    p = st.number_input("Period:", min_value=1)
                    s = sma(data, p)
                    stock_graph.add_scatter(
                        y=s["SMA"], row=i + 1, col=1, name=indicator
                    )

            case "EWMA":
                period = st.number_input("What is the period of the EWMA?", min_value=1)
                ind = ema(data, period)
                indicators.append(ind)

                count = st.checkbox("Multiple SMAs?")

                # create traces for chart
                stock_graph.add_scatter(
                    y=data["Close"],
                    mode="lines",
                    row=i + 1,
                    col=1,
                    name=data["Name"].iloc[0],
                )
                stock_graph.add_scatter(y=ind["EWMA"], row=i + 1, col=1, name=indicator)

                # check if multiple graphs are necessary
                if count:
                    p = st.number_input("Period:", min_value=1)
                    emwa = ema(data, p)
                    stock_graph.add_scatter(
                        y=emwa["EWMA"], row=i + 1, col=1, name=indicator
                    )

            case "MACD Indicator":
                ind = macd(data)
                indicators.append(ind)

                # create traces for chart
                stock_graph.add_scatter(
                    y=ind["Signal_Line"],
                    row=i + 1,
                    col=1,
                    name="Signal Line",
                )
                stock_graph.add_scatter(y=ind["MACD"], row=i + 1, col=1, name=indicator)

            case "Relative Strength Index (RSI)":
                ind = rsi(data)
                indicators.append(ind)

                # create traces for chart
                stock_graph.add_scatter(y=ind["RSI"], row=i + 1, name=indicator)

            case "Stochastic Oscillator":
                ind = stochOsc(data)
                indicators.append(ind)

                slow = st.checkbox("Slow Stochastic Oscillator?")

                # create traces for chart
                if slow:
                    stock_graph.add_scatter(
                        y=ind["%D"], row=i + 1, col=1, name="Slow Stochastic Oscillator"
                    )
                else:
                    stock_graph.add_scatter(
                        y=ind["%K"], row=i + 1, col=1, name=indicator
                    )

            case "ADX" | "Directional Movement Index":
                ind = adx(data)
                indicators.append(ind)
                # create traces for chart
                if indicator == "ADX":
                    stock_graph.add_scatter(y=ind["ADX"], row=i + 1, col=1, name="ADX")
                    stock_graph.add_scatter(
                        y=ind["ADX_smooth"], row=i + 1, col=1, name="ADX_smooth"
                    )
                else:
                    stock_graph.add_scatter(
                        y=ind["DX"], row=i + 1, col=1, name="Directional Movement Index"
                    )

            case "Average True Range":
                ind = atr(data)
                indicators.append(ind)

                # create traces for chart
                stock_graph.add_scatter(
                    y=ind["ATR"], row=i + 1, col=1, name="Average True Range"
                )

            case "Bollinger Bands":
                ind = bollinger(data)
                indicators.append(ind)

                # create traces for chart
                stock_graph.add_scatter(
                    y=data["Close"],
                    mode="lines",
                    row=i + 1,
                    col=1,
                    name=data["Name"].iloc[0],
                )
                stock_graph.add_scatter(y=ind["Upper"], row=i + 1, col=1, name="Upper")
                stock_graph.add_scatter(y=ind["Lower"], row=i + 1, col=1, name="Lower")
                stock_graph.add_scatter(y=ind["Mid"], row=i + 1, col=1, name="Mid")

            case "On Balance Volume":
                ind = obv(data)
                indicators.append(ind)

                # create traces for chart
                stock_graph.add_scatter(
                    y=ind["OBV"], row=i + 1, col=1, name="On Balance Volume"
                )

            case "Chaikin Money Flow":
                ind = chaikinMoneyFlow(data)
                indicators.append(ind)

                # create traces for chart
                stock_graph.add_scatter(
                    y=ind["CMF"], row=i + 1, col=1, name="Chaikin Money Flow"
                )

            case "True Range":
                ind = trueRange(data)
                indicators.append(ind)

                # create traces for chart
                stock_graph.add_scatter(
                    y=ind["TR"], row=i + 1, col=1, name="True Range"
                )
        i += 1
    if len(indicators) > 0:
        stock_graph.update_layout(
            title="Stock Analysis", yaxis_title="Value", height=i * 300, width=1000
        )
    else:
        stock_graph.update_layout(
            title="Stock Analysis", yaxis_title="Value", height=500, width=1000
        )

    st.plotly_chart(stock_graph)

    return indicators


sp500_stock_symbols, names, indicator_names = initialize()

# get stock symbols
stock_symbols = sp500_stock_symbols.index.get_level_values(0).unique()

st.title("Stock Analyzer")

# Separate company names and symbols and make tupel list
company_names = names["Name"].tolist()
stock_symbols = names.index.tolist()
companies = list(zip(company_names, stock_symbols))

# select the company and stock symbol
stock_name = st.selectbox(
    "Select the stock", companies, format_func=lambda x: x[0] + ", (" + x[1] + ")"
)
if "stock_name" not in st.session_state:
    st.session_state.stock_name = stock_name[0]
else:
    st.session_state.stock_name = stock_name[0]

# get stock ticker data for one stock
ticker_Data = sp500_stock_symbols.loc[
    sp500_stock_symbols.Name == st.session_state.stock_name
]

# format ticker_Data
if isinstance(ticker_Data.index, pd.MultiIndex):
    ticker_Data = ticker_Data.droplevel([0, 1])


#
#
#
st.divider()
# present ticker data
st.subheader(st.session_state.stock_name)

# option to include start and end date
# col1, col2 = st.columns(2)
# col1.date_input(
#     "Start Date",
#     ticker_Data.index.get_level_values(2)[0],
#     min_value=ticker_Data.index.get_level_values(2)[0],
#     max_value=ticker_Data.index.get_level_values(2)[-1],
# )
# col2.date_input(
#     "End Date",
#     ticker_Data.index.get_level_values(2)[-1],
#     min_value=ticker_Data.index.get_level_values(2)[0],
#     max_value=ticker_Data.index.get_level_values(2)[-1],
# )

# create chart

st.write(ticker_Data)


st.write("Choose indicators and analyze")
indicator_names = st.multiselect("Choose your indicators", indicator_names)

indicators = calculateIndicator(data=ticker_Data, indicator_names=indicator_names)
