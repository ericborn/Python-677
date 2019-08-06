# -*- coding: utf-8 -*-
"""
Eric Born
Class: CS677 - Summer 2
Date: 5 Aug 2019
Homework week 4 - stock linear regression
Create a linear regression model to predict whether to buy or sell
a stock on various window sizes
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# setup input directory and filename
ticker = 'BSX-labeled'
input_dir = r'C:\Users\TomBrody\Desktop\School\677\wk4'
ticker_file = os.path.join(input_dir, ticker + '.csv')

# read csv file into dataframe
try:
    df = pd.read_csv(ticker_file)
    print('opened file for ticker: ', ticker,'\n')

except Exception as e:
    print(e)
    print('failed to read stock data for ticker: ', ticker)

# Create separate dataframes for 2017 and 2018 data
# 2017 will be used as training, 2018 as testing for the model
df_2017 = df.loc[df['td_year']==2017]
df_2018 = df.loc[df['td_year']==2018]

# Reset indexes
df_2017 = df_2017.reset_index(level=0, drop=True)
df_2018 = df_2018.reset_index(level=0, drop=True)  

# Create position column to indicate whether its a buy or sell day.
df_2017['position'] = 0

# Creates a linear regression object
lm = LinearRegression()

# Creates a list of columns that are named by the window size that is used
# to create a dataframe containing the mean returns calculated from that
# window size.
columns = []

for i in range(5, 31):
    columns.append(str(i))

# Creates columns and stores the average return on a sliding 
# window between 5 and 30.
ret_dict = {}
for window in range(5,31):
    w = window
    p = 0
    for k in range(0, len(df_2017)):
        window_array = np.array(window)
        window_array = window_array.reshape(1, -1)
        window_mean = df_2017.iloc[p:w,-5].mean()
        window_mean = np.array(window_mean)
        window_mean = window_mean.reshape(1, -1)
        X = lm.fit(window_array, window_mean)
        ret_dict.update({[str(window)] : lm.predict(X)})
        p += 1
        w += 1

ret_dict.update({'5' : 0.01})
ret_dict.update({'6' : 0.04561})
ret_dict.update({'7' : 0.4501})
ret_dict.update({'8' : 0.04881})

for key in ret_dict.values():
    if key > 0:
        print(True)
    else:
        print(False)