# checking if libraries installed


# Importing required libraries
import yfinance as yf
import pandas as pd
from pathlib import Path
import os
import numpy as np
import time

# -----------------------------
# Define project root path
# -----------------------------
# Get the directory where this script lives, then go up to project root
# This ensures paths work regardless of where the script is called from
SCRIPT_DIR = Path(__file__).resolve().parent  # Support_Files directory
PROJECT_ROOT = SCRIPT_DIR.parent  # Project root (one level up from Support_Files)

DATA_OUTPUT = PROJECT_ROOT / "Data_Output"
DATA_OUTPUT.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Frequency configuration
# -----------------------------
VALID_FREQUENCIES = ['5m', '30m', '1h', '1d', '5d', '1wk', '1mo', '3mo']
FREQUENCY_FOLDER_MAP = {
    '5m': '5min',
    '30m': '30min',
    '1h': '1hour',
    '1d': 'daily',
    '5d': '5day',
    '1wk': 'weekly',
    '1mo': 'monthly',
    '3mo': 'quarterly'
}
DEFAULT_FREQUENCY = '1d'

# -----------------------------
# Intraday frequency limitations (yfinance constraint)
# Intraday data cannot extend past 60 days
# -----------------------------
INTRADAY_FREQUENCIES = ['5m', '30m', '1h']
INTRADAY_MAX_PERIOD = '60d'


def get_frequency_folder(frequency: str) -> str:
    """Returns the folder name for a given frequency."""
    return FREQUENCY_FOLDER_MAP.get(frequency, 'daily')


def get_output_path(ticker: str, frequency: str = DEFAULT_FREQUENCY) -> Path:
    """Returns the full output path for a ticker at a given frequency."""
    folder = get_frequency_folder(frequency)
    output_dir = DATA_OUTPUT / folder
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{ticker}.csv"


# Asking what tickers to monitor. Automatically removes duplicates and ignores case sensitivity.
# Allows commas, spaces or semicolons as separators.

# Check if tickers are in US. Remove the tickers outside of US
def ticker_in_US(tickers: str) -> list:
    valid_tickers = []
    tickers = check_ticker(tickers)
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        info = stock.info
        country = info.get("country", 0)
        if country == "United States":
            valid_tickers.append(ticker)
        else:
            print(f"Ticker: {ticker} removed as not in US according to information fetch")
    return valid_tickers


# ask interval for price data retrieval
def get_interval() -> str:
    while True:
        valid_intervals = ['1wk', '1mo', '1y', '5y', '10y', 'max']
        result = input(f"To what interval should we base our estimation? {valid_intervals}:\n")
        if result.lower().replace(" ", "") in valid_intervals:
            return result.lower().replace(" ", "")
        else:
            print("Invalid input. Please try again.")


# ask frequency for price data retrieval
def get_frequency() -> str:
    while True:
        result = input(f"What frequency for price data? {VALID_FREQUENCIES} (default: 1d):\n")
        result = result.lower().replace(" ", "")
        if result == "":
            return DEFAULT_FREQUENCY
        if result in VALID_FREQUENCIES:
            return result
        else:
            print("Invalid input. Please try again.")


def get_tickers() -> list:
    result_tickers = []
    b = True

    while b:
        tickers = input(
            "Please provide the tickers you want to monitor. Commas, spaces or semicolons can be used as separators. \nExample: AAPL, MSFT; GOOGL TSLA MSFT AMZN \n")
        separators = [',', ';', ' ']

        for separator in separators:
            tickers = tickers.replace(separator, ',')

        tickers = tickers.split(',')

        for ticker in tickers:
            ticker = ticker.upper()
            if ticker not in result_tickers:
                result_tickers.append(ticker)
        while True:
            add = input("Would you like to add another ticker? Y/N: ")
            if add.upper() == 'Y':
                b = False
                break
            elif add.upper() == 'N':
                b = True
                break
            else:
                print("Invalid input. Please try again.")

    return result_tickers


# Checking if the ticker exists on Yahoo Finance. Helps avoiding unnecessary errors later on.
def check_ticker(tickers: str) -> list:
    valid_tickers = []
    result = yf.Ticker(tickers).info
    if result["trailingPegRatio"] is None:
        print("Invalid ticker:", tickers)
        print("Ticker removed from the list.")
    else:
        valid_tickers.append(tickers)
    return valid_tickers


# Retrieve price data for the given ticker, interval, and frequency.
# Store in CSV files named after the ticker to avoid unnecessary repeated downloads.
# Important distinction: close price is the adjusted close price (accounts for dividends/splits).
# For daily frequency (1d), uses history method. For other frequencies, uses download method.
# Note: Intraday data (5m, 30m, 1h) cannot extend past 60 days per yfinance limitation.
def get_price_data(ticker: str, interval: str = "max", frequency: str = DEFAULT_FREQUENCY) -> None:
    filepath = get_output_path(ticker, frequency)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Adjust interval for intraday frequencies due to yfinance limitations
    effective_interval = interval
    if frequency in INTRADAY_FREQUENCIES and interval == "max":
        effective_interval = INTRADAY_MAX_PERIOD
        print(f"  Note: Intraday frequency '{frequency}' limited to {INTRADAY_MAX_PERIOD} of data")

    if frequency == '1d':
        # Use history method for daily data
        stock = yf.Ticker(ticker)
        prices = stock.history(period=effective_interval)
        prices = prices.reset_index()
        if 'Date' not in prices.columns and 'index' in prices.columns:
            prices = prices.rename(columns={'index': 'Date'})
    else:
        # Use download method for non-daily frequencies
        prices = yf.download(ticker, interval=frequency, auto_adjust=True, period=effective_interval)
        if isinstance(prices.columns, pd.MultiIndex):
            prices.columns = prices.columns.droplevel(1)
        prices = prices.reset_index()
        if 'Datetime' in prices.columns:
            prices = prices.rename(columns={'Datetime': 'Date'})
        elif 'index' in prices.columns:
            prices = prices.rename(columns={'index': 'Date'})

    prices.to_csv(filepath, index=False)


# Opens required csv files and calculates returns based over adjusted closing prices.
def calculate_return_close(ticker: str, frequency: str = DEFAULT_FREQUENCY) -> None:
    filepath = get_output_path(ticker, frequency)
    if not filepath.is_file():
        get_price_data(ticker, interval="max", frequency=frequency)

    prices = pd.read_csv(filepath)
    prices['Return'] = prices['Close'].pct_change()
    prices.to_csv(filepath, index=False)


# Using T-bills 5 years as risk-free rate.
# Warning: yield is expressed as integer --> 3 == 3%. Need to divide by 100 then adjust for frequency.
def get_risk_free_historical(interval: str = "max", frequency: str = DEFAULT_FREQUENCY) -> None:
    risk_free = "^FVX"
    get_price_data(risk_free, interval, frequency)


# Opens required csv file and calculate the risk free return flat.
# Flat is defined here as simple average of all time series.
def get_risk_free_daily(interval: str = "max", frequency: str = DEFAULT_FREQUENCY) -> float:
    filepath = get_output_path("^FVX", frequency)
    if not filepath.is_file():
        get_risk_free_historical(interval, frequency)
    risk_free_history = pd.read_csv(filepath)
    risk_free = risk_free_history["Close"].mean()
    return risk_free / 100 / 252  # correcting to true percentage daily


# Opens required csv files and calculates excess returns based on adjusted closing prices and t-bills average
def calculate_excess_returns_close_flat(ticker: str, frequency: str = DEFAULT_FREQUENCY) -> None:
    filepath = get_output_path(ticker, frequency)
    if not filepath.is_file():
        get_price_data(ticker, interval="max", frequency=frequency)

    prices = pd.read_csv(filepath)
    risk_free = get_risk_free_daily(frequency=frequency)

    if "Return" not in prices.columns:
        calculate_return_close(ticker, frequency)
        prices = pd.read_csv(filepath)

    prices["Excess Returns Flat"] = prices["Return"].sub(risk_free)
    prices.to_csv(filepath, index=False)


# Return daily volatility over the period as standard deviation of returns.
def calculate_volatility(ticker: str, frequency: str = DEFAULT_FREQUENCY) -> float:
    filepath = get_output_path(ticker, frequency)
    prices = pd.read_csv(filepath)
    volatility = prices['Return'].std()
    return volatility


# Retrieve industry information to become factor in regression analysis.
def get_industry(ticker: str) -> str:
    stock = yf.Ticker(ticker)
    infos = stock.info
    industry = infos.get("industry", "0")
    return industry


# Not needed, kept in case
# function to get earnings history on Yahoo Finance
def get_earnings_history(ticker: str) -> pd.DataFrame:
    stock = yf.Ticker(ticker)
    earnings = stock.earnings_history
    return earnings


# Get S&P500 data that is considered as "market" for CAPM
def get_market_data(interval: str, frequency: str = DEFAULT_FREQUENCY) -> None:
    market = "^GSPC"
    get_price_data(market, interval, frequency)


# Get all caps S&P 500 for multifactor model
def get_small_medium_large_data(interval: str, frequency: str = DEFAULT_FREQUENCY) -> None:
    market_tickers = ["^SP600", "^MID", "^GSPC"]
    for ticker in market_tickers:
        get_price_data(ticker, interval, frequency)


# retrieve factor information adjusted on Amit Goyal Investment lessons
def get_factors_data(interval: str, frequency: str = DEFAULT_FREQUENCY) -> None:
    factors = [
        "^RLV",  # Russell 1000 Value / Large Cap Value index
        "^GSPC",  # S&P 500 / Large Cap Core index
        "^RLG",  # Russell 1000 Growth / Large Cap Growth
        "MDYV",  # S&P MidCap 400 Value ETF / Mid Cap Value proxy
        "MDY",  # S&P MidCap 400 ETF / Mid Cap Core proxy
        "MDYG",  # S&P MidCap 400 Growth ETF / Mid Cap Growth proxy
        "IWN",  # iShares Russell 2000 Value / Small Cap Value
        "^RUT",  # Russell 2000 / Small Cap Core
        "IWO"  # iShares Russell 2000 Growth / Small Cap Growth
    ]
    for ticker in factors:
        get_price_data(ticker, interval, frequency)


# calculate the number of days similar outputs. If daily = 0% streak to 0
def day_streak(ticker: str, frequency: str = DEFAULT_FREQUENCY) -> None:
    filepath = get_output_path(ticker, frequency)
    if not filepath.is_file():
        get_price_data(ticker, interval="max", frequency=frequency)

    prices = pd.read_csv(filepath)

    if "Return" not in prices.columns:
        calculate_return_close(ticker, frequency)
        prices = pd.read_csv(filepath)

    return_serie = pd.Series(prices["Return"])
    serie = return_serie.tolist()

    result = []
    for i, val in enumerate(serie):
        if i == 0:
            result.append(0)
        elif val == 0 or pd.isna(val):
            result.append(0)
        elif val > 0:
            result.append(result[i - 1] + 1 if result[i - 1] > 0 else 1)
        else:
            result.append(result[i - 1] - 1 if result[i - 1] < 0 else -1)

    prices["Day_Streak"] = pd.Series(result)
    prices.to_csv(filepath, index=False)


def get_DataFrame_ready(ticker: str, frequency: str = DEFAULT_FREQUENCY) -> pd.DataFrame:
    filepath = get_output_path(ticker, frequency)

    get_price_data(ticker, interval="max", frequency=frequency)
    calculate_return_close(ticker, frequency)
    calculate_excess_returns_close_flat(ticker, frequency)
    day_streak(ticker, frequency)

    df = pd.read_csv(filepath)

    date_col = None
    if 'Date' in df.columns:
        date_col = 'Date'
    elif 'Datetime' in df.columns:
        date_col = 'Datetime'
    else:
        df = df.reset_index()
        for col in ['index', 'Date', 'Datetime']:
            if col in df.columns:
                date_col = col
                break
        if date_col is None:
            date_col = df.columns[0]

    if date_col != 'Date':
        df = df.rename(columns={date_col: 'Date'})

    df['Date'] = pd.to_datetime(df['Date'], utc=True)
    df['Date'] = df['Date'].dt.tz_localize(None)
    df['Date'] = df['Date'].dt.normalize()
    df = df.set_index('Date')

    return df


def prepare_CAPM_data(frequency: str = DEFAULT_FREQUENCY) -> pd.DataFrame:
    """
    Prepares S&P500 market data for CAPM analysis.
    Returns a DataFrame with Date as a proper column (datetime) ready for merge.
    """
    ticker = "^GSPC"
    get_market_data(interval="max", frequency=frequency)
    calculate_return_close(ticker, frequency)
    calculate_excess_returns_close_flat(ticker, frequency)

    df = pd.read_csv(get_output_path("^GSPC", frequency))

    cols_to_drop = ["Open", "High", "Low", "Volume", "Dividends", "Stock Splits"]
    df = df.drop(columns=cols_to_drop, errors='ignore')

    cols_to_change = ["Close", "Return", "Excess Returns Flat"]
    df.rename(columns={c: f"Market_{c}" for c in cols_to_change}, inplace=True)

    df['Date'] = pd.to_datetime(df['Date'], utc=True)
    df['Date'] = df['Date'].dt.tz_localize(None)
    df['Date'] = df['Date'].dt.normalize()

    print(ticker, df.head(), df.columns)
    return df


# Replicates FF3 by using ETFs (9-box approach for granular SMB/HML calculation)
def prepare_Fama_French_3(frequency: str = DEFAULT_FREQUENCY) -> pd.DataFrame:
    """
    Prepares 3-factor Fama-French market data using selected ETFs:
    - Computes market excess returns (MKT_RF)
    - Computes SMB (Small Minus Big)
    - Computes HML (High Minus Low)

    Uses 9-box style grid (3 sizes x 3 styles) for more accurate factor calculation.
    Returns DataFrame with Date as a column ready for merge.
    """
    # Define factor ETFs/indices (9-box: Size x Style)
    factors = [
        "^RLV",  # Russell 1000 Value / Large Cap Value index
        "^GSPC",  # S&P 500 / Large Cap Core index
        "^RLG",  # Russell 1000 Growth / Large Cap Growth
        "MDYV",  # S&P MidCap 400 Value ETF / Mid Cap Value proxy
        "MDY",  # S&P MidCap 400 ETF / Mid Cap Core proxy
        "MDYG",  # S&P MidCap 400 Growth ETF / Mid Cap Growth proxy
        "IWN",  # iShares Russell 2000 Value / Small Cap Value
        "^RUT",  # Russell 2000 / Small Cap Core
        "IWO"  # iShares Russell 2000 Growth / Small Cap Growth
    ]

    # Download price data and compute returns for each ETF
    print("Downloading FF3 factor data (9-box approach)...")
    df_dict = {}
    for ticker in factors:
        print(f"  Processing {ticker}...")
        try:
            get_price_data(ticker, interval="max", frequency=frequency)
            calculate_return_close(ticker, frequency)
            calculate_excess_returns_close_flat(ticker, frequency)

            m_frame = pd.read_csv(get_output_path(ticker, frequency))

            # Standardize Date column
            m_frame['Date'] = pd.to_datetime(m_frame['Date'], utc=True)
            m_frame['Date'] = m_frame['Date'].dt.tz_localize(None).dt.normalize()

            df_dict[ticker] = m_frame.set_index('Date')['Excess Returns Flat']
        except Exception as e:
            print(f"  ERROR processing {ticker}: {e}")

    # Check we have all required tickers
    missing = [t for t in factors if t not in df_dict]
    if missing:
        print(f"WARNING: Missing tickers: {missing}")

    # Compute daily excess returns aligned by date
    combined = pd.DataFrame({
        'Small_Value': df_dict.get('IWN'),
        'Small_Core': df_dict.get('^RUT'),
        'Small_Growth': df_dict.get('IWO'),
        'Mid_Value': df_dict.get('MDYV'),
        'Mid_Core': df_dict.get('MDY'),
        'Mid_Growth': df_dict.get('MDYG'),
        'Large_Value': df_dict.get('^RLV'),
        'Large_Core': df_dict.get('^GSPC'),
        'Large_Growth': df_dict.get('^RLG')
    }).dropna()

    # Compute SMB (Small Minus Big)
    # SMB = average(Small ETFs) - average(Large ETFs)
    combined['SMB'] = (
            combined[['Small_Value', 'Small_Core', 'Small_Growth']].mean(axis=1) -
            combined[['Large_Value', 'Large_Core', 'Large_Growth']].mean(axis=1)
    )

    # Compute HML (High Minus Low)
    # HML = average(Value ETFs) - average(Growth ETFs)
    combined['HML'] = (
            combined[['Small_Value', 'Mid_Value', 'Large_Value']].mean(axis=1) -
            combined[['Small_Growth', 'Mid_Growth', 'Large_Growth']].mean(axis=1)
    )

    # Compute market excess returns (MKT_RF)
    # Using S&P 500 excess return directly (already has Rf subtracted)
    combined['MKT_RF'] = combined['Large_Core']

    # Reset index so Date is a column
    combined = combined.reset_index()

    print(f"\nFF3 factors prepared:")
    print(f"  Rows: {len(combined)}")
    print(f"  Date range: {combined['Date'].min()} to {combined['Date'].max()}")
    print(combined.head())

    return combined


# Replicates FF5 by extending FF3 with RMW and CMA factors using ETFs
def prepare_Fama_French_5(frequency: str = DEFAULT_FREQUENCY) -> pd.DataFrame:
    """
    Prepares 5-factor Fama-French market data by extending FF3.

    FF5 factors:
    - MKT_RF: Market excess return (from FF3)
    - SMB: Small Minus Big (from FF3)
    - HML: High Minus Low (from FF3)
    - RMW: Robust Minus Weak (profitability factor)
    - CMA: Conservative Minus Aggressive (investment factor)

    RMW and CMA are computed using ETF proxies:
    - QUAL: iShares MSCI USA Quality Factor ETF (high profitability proxy)
    - NOBL: ProShares S&P 500 Dividend Aristocrats (conservative investment proxy)
    - MTUM: iShares MSCI USA Momentum Factor ETF (aggressive investment proxy)

    Note: ETFs start around 2013, so FF5 data range is limited compared to FF3.

    Returns DataFrame with Date as a column ready for merge.
    """
    # First get FF3 factors as base
    print("Building FF5 factors (extending FF3)...")
    ff3_df = prepare_Fama_French_3(frequency=frequency)

    if ff3_df.empty:
        print("ERROR: FF3 data is empty, cannot build FF5")
        return pd.DataFrame(columns=['Date', 'MKT_RF', 'SMB', 'HML', 'RMW', 'CMA'])

    # Set Date as index for merging
    ff3_df = ff3_df.set_index('Date')

    # Additional ETFs for RMW and CMA factors
    # These ETFs started around 2013, so FF5 will have shorter history than FF3
    rmw_cma_tickers = {
        "QUAL": "Quality",  # iShares MSCI USA Quality Factor ETF (inception: 2013)
        "NOBL": "Conservative",  # ProShares S&P 500 Dividend Aristocrats (inception: 2013)
        "MTUM": "Momentum"  # iShares MSCI USA Momentum Factor ETF (inception: 2013)
    }

    # Download and process RMW/CMA ETF data
    print("\nDownloading RMW/CMA factor ETFs...")
    etf_returns = {}
    for ticker, name in rmw_cma_tickers.items():
        print(f"  Processing {ticker} ({name})...")
        try:
            get_price_data(ticker, interval="max", frequency=frequency)
            calculate_return_close(ticker, frequency)
            calculate_excess_returns_close_flat(ticker, frequency)

            m_frame = pd.read_csv(get_output_path(ticker, frequency))

            # Standardize Date column
            m_frame['Date'] = pd.to_datetime(m_frame['Date'], utc=True)
            m_frame['Date'] = m_frame['Date'].dt.tz_localize(None).dt.normalize()

            etf_returns[ticker] = m_frame.set_index('Date')['Excess Returns Flat']
        except Exception as e:
            print(f"  ERROR processing {ticker}: {e}")

    # Check availability
    available_etfs = list(etf_returns.keys())
    print(f"  Available ETFs for RMW/CMA: {available_etfs}")

    # Need a baseline return for the "weak" profitability and "aggressive" investment
    # Using Small Cap Core (^RUT) as proxy for low profitability/high investment firms
    if 'Small_Core' in ff3_df.columns:
        baseline_return = ff3_df['Small_Core']
    else:
        baseline_return = ff3_df['MKT_RF']

    # Build combined DataFrame with FF3 + new ETFs
    combined = ff3_df.copy()

    # Add ETF returns
    for ticker in available_etfs:
        combined[ticker] = etf_returns[ticker]

    # Drop rows where any ETF data is missing (inner join effect)
    combined = combined.dropna()

    # Compute RMW (Robust Minus Weak)
    # RMW = High profitability return - Low profitability return
    # QUAL ETF holds high-quality (profitable) stocks
    # Small caps (baseline) proxy for lower profitability
    if 'QUAL' in available_etfs:
        combined['RMW'] = combined['QUAL'] - baseline_return
    else:
        print("  WARNING: QUAL not available, using HML * 0.5 as RMW proxy")
        combined['RMW'] = combined['HML'] * 0.5

    # Compute CMA (Conservative Minus Aggressive)
    # CMA = Conservative investment return - Aggressive investment return
    # NOBL holds dividend aristocrats (mature, conservative investors)
    # MTUM holds momentum stocks (often high-growth, aggressive investors)
    if 'NOBL' in available_etfs and 'MTUM' in available_etfs:
        combined['CMA'] = combined['NOBL'] - combined['MTUM']
    elif 'NOBL' in available_etfs:
        combined['CMA'] = combined['NOBL'] - combined['MKT_RF']
    else:
        print("  WARNING: NOBL/MTUM not available, using -HML * 0.3 as CMA proxy")
        combined['CMA'] = -combined['HML'] * 0.3

    # Keep only the factor columns needed
    factor_cols = ['MKT_RF', 'SMB', 'HML', 'RMW', 'CMA']

    # Also keep the 9-box columns for reference (optional)
    style_cols = ['Small_Value', 'Small_Core', 'Small_Growth',
                  'Mid_Value', 'Mid_Core', 'Mid_Growth',
                  'Large_Value', 'Large_Core', 'Large_Growth']

    cols_to_keep = factor_cols + [c for c in style_cols if c in combined.columns]
    combined = combined[cols_to_keep].copy()

    # Reset index so Date is a column
    combined = combined.reset_index()

    print(f"\nFF5 factors prepared:")
    print(f"  Rows: {len(combined)}")
    print(f"  Date range: {combined['Date'].min()} to {combined['Date'].max()}")
    print(f"  Note: Limited to ~2013+ due to ETF inception dates")
    print(combined[['Date', 'MKT_RF', 'SMB', 'HML', 'RMW', 'CMA']].head())

    return combined


def get_weekly_prices(ticker: str = "AAPL", frequency: str = "5m") -> pd.DataFrame:
    """
    Retrieves price data at specified frequency.
    Note: For intraday frequencies, yfinance has limited historical data availability
    (maximum 60 days for intraday data).
    """
    ticker = ticker.upper()
    get_price_data(ticker, interval="max", frequency=frequency)
    filepath = get_output_path(ticker, frequency)
    data = pd.read_csv(filepath)
    print(data.columns)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(data)
    return data


if __name__ == "__main__":
    start = time.time()

    list_stock = tickers = [

        "AAPL",
        "MSFT",

]
    for a in list_stock:

        get_price_data(a)
        print(a)

    get_risk_free_daily()
"""
    # Test FF3 and FF5
    print("Testing Fama-French 3 Factor Model...")
    ff3 = prepare_Fama_French_3("1d")

    print("\n" + "=" * 60)
    print("Testing Fama-French 5 Factor Model...")
    ff5 = prepare_Fama_French_5("1d")

    end = time.time()
    print(f"\nTotal code time = {end - start} seconds") """