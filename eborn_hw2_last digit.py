# -*- coding: utf-8 -*-
"""
Eric Born
Class: CS677 - Summer 2
Date: 24 July 2019
Homework week 2 - Last Digit
Compute the statistics on the distribution of the last digit (”cent” position) 
for the opening price for your stock for 5 years.
"""
import os
import pandas as pd
import numpy as np
import statistics as s
import math as m

ticker='BSX'
modifier = '-labeled'
input_dir = r'C:\Users\TomBrody\Desktop\School\677\wk2'
ticker_file = os.path.join(input_dir, ticker + modifier +'.csv')

try:
    bsx_df = pd.read_csv(ticker_file)
    print('opened file for ticker: ', ticker)
    
except Exception as e:
    print(e)
    print('failed to read stock data for ticker: ', ticker)

  
# calculate mean and standard deviation from return column
mean_return = 100.0 * bsx_df['return'].mean()
std_return = 100.0 * bsx_df['return'].std()


# setup bounds
low_bound = mean_return - 2 * std_return
upper_bound = mean_return + 2 * std_return


# Apply formatting with 2 digit precision across the open price column
bsx_df.open = bsx_df.iloc[:,7].apply(lambda x : "%.2f"%x)


# Create last_digit column inside the dataframe containing just the 
# last decimal from the open column by using -1
bsx_df['last_digit'] = bsx_df['open'].apply(lambda x: int(str(x)[-1]))


# Convert open price to float
bsx_df.open = bsx_df.open.astype(float)


# create count column, set all values to 1
bsx_df['count'] = 1


# count occurrences of each digit
digits_total = bsx_df.groupby(['last_digit'])['count'].sum()


# Calculates the percent each digit occured by multiplying digits_total
# with 100 and dividing by the total number of items in the bsx dataframe
actual = 100 * digits_total / len(bsx_df)


# Create array of 10 10's to represent the likelihood of each digit
predicted = np.array([10,10,10,10,10,10,10,10,10,10])


# Mean squared error is the mean of actual minus predicted squared
mse = np.mean((actual - predicted)**2)


# 1)
# Most frequent digit
print('The most frequent digit is:', 
      digits_total[digits_total == max(digits_total)].index[0],
      'occurring', max(digits_total), 'times')


# 2)
# least frequent digit
print('The least frequent digit is:', 
      digits_total[digits_total == min(digits_total)].index[0],
      'Occuring', min(digits_total), 'times')


# 3)
# a)
print('The max absolute error is:', round(max(actual - predicted), 4))

# b)
print('The median absolute error is:', round(s.median(actual - predicted), 4))

# c)
print('The mean absolute error is:', round(s.mean(actual - predicted), 4))

# d)
print('The root mean squared error is:', round(m.sqrt(mse), 4))