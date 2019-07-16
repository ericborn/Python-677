# -*- coding: utf-8 -*-
"""
@course: MET CS 677
@date: July 2019
@author: epinsky
@description: Read and Save Stock date from Yahoo Finance
Notes:
Running this program requires the pandas and pandas_datareadermodules, which
may not be included with your python build. Additional modules can be easily
added using "PIP".
PIP is a recursive acronym that stands for “PIP Installs Packages” or
“Preferred Installer Program”.
Pandas should already be installed with your Anaconda build.
To install datareader, from the command line run:
   pip install pandas_datareader
* It may take 30 seconds for the command to resolve and start the install
"""

import os
import pandas as pd
from pandas_datareader import data as web

def get_stock(ticker, start_date, end_date, s_window, l_window):
    '''
    Fetch stock data from Yahoo Finance and return a pnadas dataframe.
    Arguments:
       ticker: stock ticker (string)
            Note that get_data_yahoo() supports array-like object (list, tuple,
            Series), Single stock symbol (ticker), array-like
            object of symbols or DataFrame with index containing stock symbols.
       start_date: string, (defaults to '1/1/2010')
             Starting date, timestamp.
             Parses many different kind of date representations
             (e.g., 'JAN-01-2010', '1/1/10', 'Jan, 1, 1980')
       end_date: string, (defaults to today)
             Ending date, timestamp. Same format as starting date.
       s_window: Rolling Short Moving Average Window (int)
       l_window: Rolling Long Moving Average Window (int)
    Output Dataframe:
       trade_date: Trading date
       td_year: Year of trading date
       td_month: Month of trading date
       td_day: Day of trading date
       td_weekday: Weekday of trading date
       td_week_number: Week # of trading date
       td_year_week: Year-Week # of trading date
       open: : Trading day openning price of stock
       high: Trading day high price
       low: Trading day low price
       close: Trading day high price
       volume: Trading day valume
       adj_close: Trading day divident adjusted closing price
       return: Daily Return (using adj_close)
       short_ma: Rolling Short Moving Average
       long_ma: Rolling Long Moving Average
    Example:
    df = web.get_data_yahoo('AAPL', start='2018-01-01', end='2018-04-30')
    df = web.get_data_yahoo('AAPL', start='1/1/2019')
    '''

    # Fetch stock data from Yahoo Finance
    try:
        df = web.get_data_yahoo(ticker, start=start_date, end=end_date)
    except Exception as e:
        print('ERROR:', e)
        return None

    # Update column names to be PEP 8 compliant and round to 2 decimals.
    df.rename(columns={'Adj Close':'adj_close'}, inplace=True)
    for col in list(df.columns):
        df[col] = df[col].round(2)
        df.rename(columns={col:col.lower()},inplace=True)

    # Create column from date index
    df['trade_date'] = df.index
    df['trade_date'] = pd.to_datetime(df['trade_date'])

    # Create columns from date
    df['td_year'] = df['trade_date'].dt.year
    df['td_month'] = df['trade_date'].dt.month
    df['td_day'] = df['trade_date'].dt.day
    df['td_weekday'] = df['trade_date'].dt.weekday_name
    df['td_week_number'] = df['trade_date'].dt.strftime('%U')
    df['td_year_week'] = df['trade_date'].dt.strftime('%Y-%U')

    # Calculate additional columns
    df['return'] = df['adj_close'].pct_change()
    df['return'].fillna(0, inplace=True)
    df['short_ma'] = df['adj_close'].rolling(window=s_window, min_periods=1).\
        mean()
    df['long_ma'] = df['adj_close'].rolling(window=l_window, min_periods=1).\
        mean()

    col_list = ['trade_date', 'td_year', 'td_month', 'td_day', 'td_weekday',
                'td_week_number', 'td_year_week', 'open',
                'high', 'low', 'close', 'volume', 'adj_close',
                'return', 'short_ma', 'long_ma']
    df = df[col_list]
    print('Read {:,.0f} lines of data for ticker: {}'.format(
        len(df), ticker))
    return df

if __name__ == '__main__':
    # Ticker to fetch
    ticker='BSX'
    # Input directory (currently set to be the run directory)
    input_dir = '.'
    # Create an output file name
    output_file = os.path.join(input_dir, ticker + '.csv')
    # Fetch the stock data
    df = get_stock(ticker, start_date='2014-01-01', end_date='2018-12-31',
               s_window=14, l_window=50)
    assert (df is not None and len(df) > 0), (
        'Failed to get Yahoo stock data for ticker: {}'.format(ticker))
    # Write the stock data to a file
    # Note that there is no checking whether the csv file already exists!
    df.to_csv(output_file, index=False)


