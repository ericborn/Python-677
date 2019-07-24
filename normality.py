# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 17:06:52 2019

@author: Eric Born
"""

import os
import pandas as pd
import statistics as s

ticker = 'BSX'
input_dir = r'C:\Users\TomBrody\Desktop\school\677\wk2'
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

# store total number of trading days per year
trade_days = bsx_df.groupby('td_year')['td_year'].value_counts()

# use dictionaries to store year and total positive/negative trading days
years_p = {2013: 1, 2014: 1, 2015: 1, 2016: 1, 2017:1, 2018:1}
years_n = {2013: 1, 2014: 1, 2015: 1, 2016: 1, 2017:1, 2018:1}

for j in range(2013, 2019):
    for i in range(1, len(bsx_df) - 1):
        if all((bsx_df.iloc[[i], 1] == j) & all(bsx_df.iloc[[i], -3] > 0)):
            years_p[j] += 1

        if all((bsx_df.iloc[[i], 1] == j) & all(bsx_df.iloc[[i], -3] <= 0)):
            years_n[j] += 1
print('number of days with positive returns: ' + str(years_p))
print('number of days with negative returns: ' + str(years_n))

# 2)
# average returns per year            
means = dict(round(bsx_df.groupby('td_year')['returns'].mean(), 4))

# use dictionaries to store year and total above/below average trading days
years_above_avg = {2013: 1, 2014: 1, 2015: 1, 2016: 1, 2017:1, 2018:1}
years_below_avg = {2013: 1, 2014: 1, 2015: 1, 2016: 1, 2017:1, 2018:1}

# Update dictionaries with average stats
for key in means:
    for i in range(1, len(bsx_df) - 1):
        if all((bsx_df.iloc[[i], 1] == key) & all(bsx_df.iloc[[i], -3] > means[key])):
            years_above_avg[key] += 1
        
        if all((bsx_df.iloc[[i], 1] == key) & all(bsx_df.iloc[[i], -3] < means[key])):
            years_below_avg[key] += 1

# Print stats by year, total trading days, average return, days
print('Year ' + 'Trading days ' + '    µ ' + '    %days<µ ' + '%days>µ')
for year in years_above_avg.keys():
    print(year, '    ', trade_days[year].values, '    ', means[year],
          '    ',int(100 * round(years_above_avg[year] / trade_days[year], 2)), 
          '    ',int(100 * round(years_below_avg[year] / trade_days[year], 2)))
          
# 3)
# store standard devation * 2 for comparison against daily returns
st_devs = dict(round(bsx_df.groupby('td_year')['returns'].std(), 6))

# store lower and upper deviation bounds by year
lower_dev = {k: means[k] - (st_devs[k] * 2) for k in means}
upper_dev = {k: means[k] + (st_devs[k] * 2) for k in means}

# Create dictionaries to hold how many days above/below 2 standard deviations
years_above_dev = {2013: 1, 2014: 1, 2015: 1, 2016: 1, 2017:1, 2018:1}
years_below_dev = {2013: 1, 2014: 1, 2015: 1, 2016: 1, 2017:1, 2018:1}

# Update the dictionaries
for key in means:
    for i in range(1, len(bsx_df) - 1):
        if all((bsx_df.iloc[[i], 1] == key) & 
           all(bsx_df.iloc[[i], -3] > upper_dev[key])):
            years_above_dev[key] += 1
        
        if all((bsx_df.iloc[[i], 1] == key) & 
           all(bsx_df.iloc[[i], -3] < lower_dev[key])):
            years_below_dev[key] += 1
# Output days greater or less than 2 standard deviations from the mean
print('Total number of days greater than 2 standard deviations from the mean:')
print(years_above_dev)
print('\nTotal number of days less than 2 standard deviations from the mean:')
print(years_below_dev)

# 4)
print('Year ' + 'Trading days ' + '    µ ' + '     σ' +
      '        %days<µ ' + '%days>µ')
for year in years_above_dev.keys():
    print(year, '    ', trade_days[year].values, '    ', means[year], st_devs[year],
          '    ',int(100 * round(years_above_dev[year] / trade_days[year], 2)), 
          '    ',int(100 * round(years_below_dev[year] / trade_days[year], 2)))

          