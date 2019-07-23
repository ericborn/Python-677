# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 17:06:52 2019

@author: Eric Born
"""

import os
import pandas as pd

ticker = 'BSX'
input_dir = r'C:\Users\eborn\Desktop\school\677\wk2'
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

# explicitly set column types for ease in calculating
bsx_df.td_year = bsx_df.td_year.astype(int)
bsx_df.returns = bsx_df.returns.astype(float)

# use dictionary to store year and total positive negative instead of list
years_p = {2013: 1, 2014: 1, 2015: 1, 2016: 1, 2017:1, 2018:1}
years_n = {2013: 1, 2014: 1, 2015: 1, 2016: 1, 2017:1, 2018:1}

for j in range(2013, 2019):
    for i in range(1, len(bsx_df) - 1):
        if all((bsx_df.iloc[[i], 1] == j) & all(bsx_df.iloc[[i], -3] > 0)):
            years_p[j] += 1

        if all((bsx_df.iloc[[i], 1] == j) & all(bsx_df.iloc[[i], -3] < 0)):
            years_n[j] += 1

# average returns per year            
means = dict(round(bsx_df.groupby('td_year')['returns'].mean(), 4))

years_above_avg = {2013: 1, 2014: 1, 2015: 1, 2016: 1, 2017:1, 2018:1}
years_below_avg = {2013: 1, 2014: 1, 2015: 1, 2016: 1, 2017:1, 2018:1}

for key in means:
    for i in range(1, len(bsx_df) - 1):
        if all((bsx_df.iloc[[i], 1] == key) & all(bsx_df.iloc[[i], -3] > means[key])):
            years_above_avg[key] += 1
        
        if all((bsx_df.iloc[[i], 1] == key) & all(bsx_df.iloc[[i], -3] < means[key])):
            years_below_avg[key] += 1

for year in years_above_avg.keys():
    print(year, years_above_avg[year] + years_below_avg[year], means[year],
          years_above_avg[year], years_below_avg[year])
