# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 11:08:49 2019

@author: Eric Born
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 17:38:54 2019

@author: Eric Born
"""

import os
import pandas as pd

ticker = 'BSX-labeled'
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
        'volume', 'adj_close', 'return,', 'short_ma', 'long_ma', 'label']

# Create dataframe
bsx_df = pd.DataFrame([sub.split(',') for sub in lines], columns = cols)

# Drop the first row which contains the header since the column names are
# set during the dataframe construction
bsx_df = bsx_df.drop([0], axis = 0)

# Convert open price to float
bsx_df.open = bsx_df.open.astype(float)

# Apply formatting across the open price column to
bsx_df.iloc[:,7] = bsx_df.iloc[:,7].apply(lambda x : "%.2f"%x)

zero = 0
one = 0
two = 0
three = 0
four = 0
five = 0
six = 0
seven = 0
eight = 0
nine = 0

for i in range(len(bsx_df) -1):
    if bsx_df.iloc[i,7][-1] == '0':
        zero = zero + 1
    if bsx_df.iloc[i,7][-1] == '1':
        one = one + 1
    if bsx_df.iloc[i,7][-1] == '2':
        two = two + 1
    if bsx_df.iloc[i,7][-1] == '3':
        three = three + 1
    if bsx_df.iloc[i,7][-1] == '4':
        four = four + 1
    if bsx_df.iloc[i,7][-1] == '5':
        five = five + 1
    if bsx_df.iloc[i,7][-1] == '6':
        six = six + 1
    if bsx_df.iloc[i,7][-1] == '7':
        seven = seven + 1
    if bsx_df.iloc[i,7][-1] == '8':
        eight = eight + 1
    if bsx_df.iloc[i,7][-1] == '9':
        nine = nine + 1    

max(zero, one, two, three, four, five, six, seven, eight, nine)
