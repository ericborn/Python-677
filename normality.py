# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 17:06:52 2019

@author: Eric Born
"""

import os
import pandas as pd

ticker = 'BSX'
input_dir = r'C:\Users\TomBrody\Desktop\School\677\wk2'
ticker_file = os.path.join(input_dir, ticker + '.csv')

try:
    with open(ticker_file) as f:
        lines = f.read().splitlines()
        # list_lines = lines.split('\n')
    print('opened file for ticker: ', ticker)
    """    your code for assignment 1 goes here
    """

except Exception as e:
    print(e)
    print('failed to read stock data for ticker: ', ticker)


# Setup column names for the dataframe
cols = ['trade_date', 'td_year', 'td_month', 'td_day', 'td_weekday',
        'td_week_number', 'td_year_week', 'open', 'high', 'low', 'close', 
        'volume', 'adj_close', 'returns', 'short_ma', 'long_ma']

# Create dataframe
bsx_df = pd.DataFrame([sub.split(',') for sub in lines], columns = cols)

# Drop the first row which contains the header since the column names are
# set during the dataframe construction
bsx_df = bsx_df.drop([0], axis = 0)

yr_13_p = 0
yr_14_p = 0
yr_15_p = 0
yr_16_p = 0
yr_17_p = 0
yr_18_p = 0

yr_13_n = 0
yr_14_n = 0
yr_15_n = 0
yr_16_n = 0
yr_17_n = 0
yr_18_n = 0

# explicitly set column types for ease in calculating
bsx_df.td_year = bsx_df.td_year.astype(int)
bsx_df.returns = bsx_df.returns.astype(float)

(bsx_df.iloc[[1], -3].item() > 0) == True

bsx_df.iloc[[0], 1].item() == 2013

bsx_df[[1], -3]


for i in range(1, len(bsx_df) - 1):
    if bsx_df.all((bsx_df.iloc[[i], 1].item() == 2013) & (bsx_df.iloc[[i], -3] > 0)):
        yr_13_p = yr_13_p + 1

    if ((bsx_df.iloc[[i], 1].item() == 2013) & (bsx_df.iloc[[i], -3] < 0)):
        yr_13_n = yr_13_n + 1
    
