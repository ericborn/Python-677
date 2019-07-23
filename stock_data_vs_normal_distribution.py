# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 14:37:29 2018

@author: epinsky
"""
# run this  !pip install pandas_datareader
from pandas_datareader import data as web
import os
import math
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

ticker='BSX'
input_dir = r'C:\Users\TomBrody\Desktop\School\677\wk2'
ticker_file = os.path.join(input_dir, ticker + '.csv')
plot_dir = r'C:\Users\TomBrody\Desktop\School\677\wk2'

try:   
    df = pd.read_csv(ticker_file)
    df['return'] = 100.0 * df['return']
    year = '2016'
    start_date=year + '-01-01'; 
    end_date=year + '-12-31'
    df = df[df['trade_date'] >= start_date]
    df = df[df['trade_date'] <= end_date]
    low_return = -5
    high_return = 5
    df = df[(df['return']>low_return) & (df['return'] < high_return)]
    fig = plt.figure()
    returns_list = df['return'].values
    plt.hist(returns_list, normed=True, bins = 30, label='Daily Returns')
    x = np.linspace(low_return, high_return, 1000)
    pos = len(df[df['return'] > 0])
    neg = len(df[df['return'] < 0])
    ticker_mean = df['return'].mean()
    ticker_std = df['return'].std()

    plt.plot(x, mlab.normpdf(x, ticker_mean,ticker_std), color='red',
         label='Normal, ' + r'$\mu=$' + str(round(ticker_mean,2)) +  
         ', ' + r'$\sigma=$' + str(round(ticker_std,2)))
    plt.title('daily returns for ' + ticker +  ' for year ' +  year)
    plt.legend()
    output_file = os.path.join(plot_dir, 'returns_' + year + '.pdf')
    plt.savefig(output_file)

    
except Exception as e:
    print(e)
    print('failed to read stock data for ticker: ', ticker)













