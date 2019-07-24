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

#import os
import pandas as pd
import numpy as np
import statistics as s
import math as m

input_dir = r'C:\Users\TomBrody\Desktop\School\677\wk2\BSX-labeled.csv'

bsx_df = pd.read_csv(input_dir)

# mean and std return
mean_return = 100.0 * bsx_df['return'].mean()
std_return = 100.0 * bsx_df['return'].std()

# setup bounds
low_bound = mean_return - 2 * std_return
upper_bound = mean_return + 2 * std_return

# Apply formatting with 2 digit precision across the open price column
bsx_df.open = bsx_df.iloc[:,7].apply(lambda x : "%.2f"%x)
bsx_df['last_digit'] = bsx_df['open'].apply(lambda x: int(str(x)[-1]))

# Convert open price to float
bsx_df.open = bsx_df.open.astype(float)

# create count column, set all values to 1
bsx_df['count'] = 1

# count occurrences of each digit
digits_total = bsx_df.groupby(['last_digit'])['count'].sum()
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

