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
df_returns = pd.DataFrame()
ret_dict = {}
t = 0
for window in range(5,6):
    #stock_mean = []
    w = window
    p = 0
    for k in range(0, len(df_2017)):
        X = np.array(df_2017.iloc[p:w, -6]).reshape(-1, 1)
        lm.fit(X, df_2017.iloc[p:w, -8])
        
        if w == 250:
            break
        else:
            pred = lm.predict(np.array(df_2017.iloc[w+1, -6]).reshape(-1, 1))
        
        if float(pred) > df_2017.iloc[w, -6]:
            df_2017.iloc[w, -1] = 1
        else:
            df_2017.iloc[w, -1] = -1
        p += 1
        w += 1
    #df_returns[columns[t]] = stock_mean
    #t += 1

# Initialize wallet and shares to track current money and number of shares.
wallet = 100.00
shares = 0
worth = 0










def estimate_coef (x, y):
    n = np.size (x)
    mu_x, mu_y = np.mean(x), np.mean(y)
    SS_xy = np.sum(y * x) - n * mu_y * mu_x
    SS_xx = np.sum(x * x) - n * mu_x * mu_x
    slope = SS_xy / SS_xx
    intercept = mu_y - slope * mu_x
    return (slope, intercept)

def plot_regression (x, y, slope, intercept):
    plt.scatter (x, y, color = 'blue',
                 marker = 'o', s = 100)
    y_pred = slope * x + intercept
    plt.plot (x, y_pred, color = 'green', lw = 3)
    plt.xlabel ('x')
    plt.ylabel ('y')
    plt.show ()




lm.fit(X, df_2017.iloc[0:5,-8])

    
# Creates columns and stores the average return on a sliding 
# window between 5 and 30.
df_returns = pd.DataFrame()
stock_mean = []
t = 0
for i in range(5,31):
    stock_mean = []
    w = i
    p = 0
    for k in range(0, len(df_2017)):
        lm.fit(i, df_2017.iloc[p:w,-5])
        #stock_mean.append(round(df_2017.iloc[p:w,-5], 4))
        p += 1
        w += 1
    #df_returns[columns[t]] = stock_mean
    #t += 1    
    
    

df_returns.columns

df_2017.iloc[:,-5]

# Create a list containing values for each window size and yearly 
# average returns
df_avg = []
for i in range(0, len(columns)):
    df_avg.append([columns[i], round(df_returns.iloc[:,i].mean(), 8)])
    #df_avg.append({columns[i]: round(df_returns.iloc[:,i].mean(), 8)})


  
# create
for i in range(len(df_avg)):
    plt.scatter(df_avg[i][0], df_avg[i][1], color='blue')
plt.ylim(-0.0001, 0.0005)
plt.show()
    
# Create dataframe that holds columns relating to the stocks daily activity
X = df_2017.iloc[:,np.r_[7:13, 14:16]]

# Creates a linear regression object
lm = LinearRegression()

# Fit the model using stock activity inside X and the adjusted closing price
lm.fit(X, df_avg.iloc[:,0])

lm.predict(X)

coeff_df = pd.DataFrame({'features': X.columns.values,
                        'estimatedCoefficient': lm.coef_})

plt.scatter(df_2017.returns, df_2017.long_ma, color='red')
plt.xlabel('long_ma')
plt.ylabel('return')
plt.title('long_ma vs. return')
plt.show()
    

# first 5 predicted prices
lm.predict(X)[:5]   

# Compare predicted against real returns
plt.scatter(df_avg.iloc[:,0]* 1000, lm.predict(X) * 1000)
plt.xlabel('returns')
plt.ylabel('predicted returns')
plt.title('Real vs. Predicted returns')
plt.show()




# 5)
# Implemented trading strategy based upon label predicitons vs
# buy and hold strategy

# Initialize wallet and shares to track current money and number of shares.
wallet = 100.00
shares = 0
worth = 0

# stores adj_close values for the last day of each trading week
adj_close = df_2018.groupby('td_week_number')['adj_close'].last()

# stores open price for the first day of each trading week
open_price = df_2018.groupby('td_week_number')['open'].first()

# for loop that evaluates the dataset deciding when to buy/sell based
# upon the prediction labels. 0 is a bad week, 1 is a good week
try:
    for i in range(0, len(pred_2018)):
        # Sell should occur on the last day of a green week at 
        # the adjusted_close price. Since i is tracking the current
        # trading week we need to minus 1 to get the adjusted close price
        # from the previous trading week
        if pred_2018[i] == 0 and shares > 0:
            wallet = round(shares * adj_close[i - 1], 2)
            shares = 0
            
        # Buy should occur on the first day of a green week at the open price
        if pred_2018[i] == 1 and shares == 0: 
            shares = wallet / open_price[i]
            wallet = 0            
            
except Exception as e:
    print(e)
    print('Failed to evaluate df_2018 labels')


# set worth by multiplying stock price on final day by total shares
worth = round(shares * adj_close[52], 2)

if worth == 0:
    worth = wallet
    profit = round(wallet - 100.00, 2)
else:
    profit = round(worth - 100.00, 2)

# Total Cash: $0
# Total shares: 6.703067 
# Worth: $236.89
# This method would close the year at $ 141.7 a profit of $ 41.7
print('\n2018 Label Strategy:')
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
shares = round(wallet / float(open_price[0]), 6)
worth = round(shares * adj_close[52], 2)
profit = round(worth - 100.00, 2)

#Currently own 4.009623 shares 
#Worth $ 141.70
#Selling on the final day would result in $ 141.7 a profit of $ 41.7
print('\n2018 buy and hold:','\nCurrently own', shares, 'shares',
      '\nWorth','$',"%.2f"%round(worth, 2))
print('Selling on the final day would result in $',"%.2f"%worth, 'a profit of $', "%.2f"%profit)

