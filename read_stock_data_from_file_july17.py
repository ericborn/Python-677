# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 14:37:29 2018

@author: epinsky
this scripts reads your ticker file (e.g. MSFT.csv) and
constructs a list of lines
"""
import os
import math
import numpy as np
import pandas as pd

#
ticker='MSFT'
input_dir = r'C:\Users\epinsky\bu\python\data_science_with_Python\datasets'
ticker_file = os.path.join(input_dir, ticker + '.csv')
output_file = os.path.join(input_dir, ticker + '_updated.csv')

df = pd.read_csv(ticker_file)

mean_return = 100.0 * df['Return'].mean()
std_return = 100.0 * df['Return'].std()

low_bound = mean_return - 2 * std_return
upper_bound = mean_return + 2 * std_return


# please fix the computation of last digit
df['Open'] = df['Open'].round(2)
df['last_digit'] = df['Open'].apply(lambda x: int(str(x)[-1]))
df['count'] = 1

digits_total = df.groupby(['last_digit'])['count'].sum()
actual = 100 * digits_total / len(df)

predicted = np.array([10,10,10,10,10,10,10,10,10,10])

mse = np.mean((actual - predicted)**2)








