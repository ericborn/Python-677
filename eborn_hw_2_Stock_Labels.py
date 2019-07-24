# -*- coding: utf-8 -*-
"""
Eric Born
Class: CS677 - Summer 2
Date: 24 July 2019
Homework week 2 - Stock Labels
Script that evaluates buy-and-hold vs utilizing green/red labels for calculating
when to buy and sell a stock and measuring their performance.
"""

import os
import pandas as pd

ticker = 'BSX-labeled'
input_dir = r'C:\Users\TomBrody\Desktop\School\677\wk2\Labels'
ticker_file = os.path.join(input_dir, ticker + '.csv')

try:
    with open(ticker_file) as f:
        lines = f.read().splitlines()
        # list_lines = lines.split('\n')
    print('opened file for ticker: ', ticker)

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

# Initialize wallet and shares to track current money and number of shares.
wallet = 100.00
shares = 0
worth = 0

# range 0-253 is the first year (2014) with labels
# -1 in the if statement is the label column, -5 is the adj_close column.
# i - 1 and -5 sells and buys on the adjusted close from the previous day
# Using .item() to return only the value, not the name or dtype.
try:
    for i in range(0, 253):
        # SELL
        if bsx_df.iloc[[i], -1].item() == 'R' and shares > 0:
            wallet = round(shares * float(bsx_df.iloc[[i - 1], -5].item()), 2)
            shares = 0
        
        # Buy
        if bsx_df.iloc[[i], -1].item()  == 'G' and shares == 0:
            shares = wallet / float(bsx_df.iloc[[i - 1], -5].item())
            wallet = 0
except Exception as e:
    print(e)
    print('Failed to evaluate bsx_dataframe labels for 2014')

# set worth by multiplying stock price on final day by total shares
worth = round(float(bsx_df.iloc[[253], -7].item()) * shares, 2)

if worth == 0:
    worth = wallet
    profit = round(wallet - 100.00, 2)
else:
    profit = round(worth - 100.00, 2)

# Total Cash: $0
# Total shares: 14.487179487179487
# Worth: $200.07
print('\n2014 Label Strategy:')
print('Total Cash: $', wallet, '\nTotal shares:', round(shares, 6),
      '\nWorth: $', worth)    
print('This method would close the year at $', worth, 'a profit of $', profit)

# Buy and hold
# Initialize wallet and shares to track current money and number of shares.
wallet = 100.00
shares = 0
profit = 0
worth = 0

# Calculate shares, worth and profit
shares = round(wallet / float(bsx_df.iloc[[0], 7].item()), 6)
worth = round(shares * float(bsx_df.iloc[[253], -5].item()), 2)
profit = round(worth - 100.00, 2)

print('\n2014 buy and hold:','\nCurrently own', shares, 'shares',
      '\nWorth','$',round(worth, 2))
print('Selling on the final day would result in $', worth, 'a profit of $', profit)

# Initialize wallet and shares to track current money and number of shares.
wallet = 100.00
shares = 0
worth = 0
profit = 0

# range 756-1006 is the second year (2017) with labels
# -1 in the if statement is the label column, -5 is the adj_close column.
# i - 1 and -5 sells and buys on the adjusted close from the previous day
# Using .item() to return only the value, not the name or dtype.
try:
    for i in range(756, 1006):
        # SELL
        if bsx_df.iloc[[i], -1].item() == 'R' and shares > 0:
            wallet = round(shares * float(bsx_df.iloc[[i - 1], -5].item()), 2)
            shares = 0
        
        # Buy
        if bsx_df.iloc[[i], -1].item()  == 'G' and shares == 0:
            shares = wallet / float(bsx_df.iloc[[i - 1], -5].item())
            wallet = 0
except Exception as e:
    print(e)
    print('Failed to evaluate bsx_dataframe labels for 2017')

# set worth by multiplying stock price on final day by total shares
worth = round(float(bsx_df.iloc[[1006], -7].item()) * shares, 2)

if worth == 0:
    worth = wallet
    profit = round(wallet - 100.00, 2)
else:
    profit = round(worth - 100.00, 2)
    
# Total Cash: $157.48
# Total shares: 0
# Worth: $0.0
print('\n2017 Label Strategy:')
print('Total Cash: $', wallet, '\nTotal shares:', round(shares, 6),
      '\nWorth: $', worth)    
print('This method would close the year at $', worth, 'a profit of $', profit)

# Buy and hold
# Initialize wallet and shares to track current money and number of shares.
wallet = 100.00
shares = 0
hold_worth = 0
hold_stocks = 0

# Calculate shares, worth and profit
shares = round(wallet / float(bsx_df.iloc[[756], 7].item()), 6)
worth = round(shares * float(bsx_df.iloc[[1006], -5].item()), 2)
profit = round(worth - 100.00, 2)

print('\n2017 buy and hold: \n' + 'Currently own ' + str(hold_stocks) + ' shares' + '\n' +
      'Worth ' + '$' + str(round(hold_worth, 2)))
print('Selling on the final day would result in $', worth, 'a profit of $', profit)


# Initialize wallet and shares to track current money and number of shares.
wallet = 100.00
shares = 0
worth = 0
profit = 0

# range 1007-len(bsx_df) is the third year (2018) with labels
# -1 in the if statement is the label column, -5 is the adj_close column.
# i - 1 and -5 sells and buys on the adjusted close from the previous day
# Using .item() to return only the value, not the name or dtype.
try:
    for i in range(1007, len(bsx_df) - 1):
        # SELL
        if bsx_df.iloc[[i], -1].item() == 'R' and shares > 0:
            wallet = round(shares * float(bsx_df.iloc[[i - 1], -5].item()), 2)
            shares = 0
        
        # Buy
        if bsx_df.iloc[[i], -1].item()  == 'G' and shares == 0:
            shares = wallet / float(bsx_df.iloc[[i - 1], -5].item())
            wallet = 0       
except Exception as e:
    print(e)
    print('Failed to evaluate bsx_dataframe labels for 2018')

# set worth by multiplying stock price on final day by total shares
worth = round(float(bsx_df.iloc[[-1], -7].item()) * shares, 2)

if worth == 0:
    worth = wallet
    profit = round(wallet - 100.00, 2)
else:
    profit = round(worth - 100.00, 2)
         
# Total Cash: $0
# Total shares: 5.714285714285714
# Worth: $78.91
print('\n2018 Label Strategy:')
print('Total Cash: $', wallet, '\nTotal shares:', round(shares, 6),
      '\nWorth: $', worth)    
print('This method would close the year at $', worth, 'a profit of $', profit)

# Buy and hold
# Initialize wallet and shares to track current money and number of shares.
wallet = 100.00
shares = 0
hold_worth = 0
hold_stocks = 0

# Calculate shares, worth and profit
shares = round(wallet / float(bsx_df.iloc[[1007], 7].item()), 6)
worth = round(shares * float(bsx_df.iloc[[-1], -5].item()), 2)
profit = round(worth - 100.00, 2)

print('\n2018 buy and hold: \n' + 'Currently own ' + str(hold_stocks) + ' shares' + '\n' +
      'Worth ' + '$' + str(round(hold_worth, 2)))
print('Selling on the final day would result in $', worth, 'a profit of $', profit)