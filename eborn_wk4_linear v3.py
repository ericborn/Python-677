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
import sys
from sklearn.linear_model import LinearRegression

# Set display options for dataframes
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 50)

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
    sys.exit('failed to read stock data for ticker: '+ str(ticker))

# Create separate dataframes for 2017 and 2018 data
# 2017 will be used as training, 2018 as testing for the model
df_2017 = df.loc[df['td_year']==2017]
df_2018 = df.loc[df['td_year']==2018]

# Reset indexes
df_2017 = df_2017.reset_index(level=0, drop=True)
df_2018 = df_2018.reset_index(level=0, drop=True)  

# Create position column to indicate whether its a buy or sell day.
#df_2017['position'] = 0

# Creates a linear regression object
lm = LinearRegression()

# stores the position 0, 1, -1 for each window size
position_df  = pd.DataFrame()

# window = number of days to evalute before making a prediction values 5-30
# adj_close price is used to train the regression model
# close price is being predicted
# window_end = window - 1 = total size of the window
# window_start = start of the window
# adj_close = array of adj_close prices inside window (x axis)
# close = array of close prices inside window (y axis)
for window in range(5,31):
    # Create position column to indicate whether its a buy or sell day.
    # column is reset to all 0's at the start of each loop iteration
    df_2017['position'] = 0
    
    # set window_end equal to window - 1 due to zero index
    window_start = 0
    window_end = window - 1
    
    # loop that handles gathering the adj_close and close price 
    # for the appropriate window size
    for k in range(0, len(df_2017)):
        adj_close = np.array(df_2017.loc[window_start:window_end,
                                 'adj_close']).reshape(-1, 1)
        close = np.array(df_2017.loc[window_start:window_end,
                                 'close']).reshape(-1, 1)
        lm.fit(adj_close, close)
        
        # Breaks on the last row since it cannot predict w + 1 if 
        # there is no data for the next day, else it creates
        # a prediction.
        if window_end == len(df_2017) - 1:
            break
        else:
            pred = lm.predict(np.array(df_2017.loc[window_end + 1, 
                                      'adj_close']).reshape(-1, 1))
        
        # updates the position column with a 1 when prediciton for tomorrows
        # close price (w + 1) is greater than the close price of w.
        # Else it marks it with a -1 to indicate a lower price.
        if float(pred) >= df_2017.loc[window_end, 'close']:
            df_2017.loc[window_end, 'position'] = 1
        else:
            df_2017.loc[window_end, 'position'] = -1
        window_start += 1
        window_end += 1
    
    # writes the position column to a the position dataframe after each
    # window iteration
    position_df[str(window)] = df_2017.loc[:, 'position']
    
# Initialize wallet and shares to track current money and number of shares.
#wallet = 100.00
long_shares = 0
long_worth = 0
long_price = 0
short_shares = 0
short_worth = 0
short_price = 0

# Declare column names and dataframe for storing stock sales information
col_names = ['long_price','long_shares', 'long_worth', 'short_price', 
             'short_shares', 'short_worth']
stocks_df = pd.DataFrame(columns = col_names)

# Manual variable setters
#position_df.iloc[:, 0]
position_column = 0
position_row = 4

# for loop that evaluates the dataset deciding when to buy/sell based
# upon the prediction labels. 0 is a bad week, 1 is a good week
try:
    for position_column in range(0, len(position_df.iloc[0, :])):
        for position_row in range(0, len(position_df)):
            
            # Buy section
            # long_shares buy should occur if position dataframe  
            # contains a 1 and there are no long_shares held
            if (position_df.iloc[position_row, position_column] == 1 
            and long_shares == 0): 
                long_shares = 100.00 / df_2017.loc[position_row, 'close']           
                long_price = df_2017.loc[position_row, 'close']
                long_worth = 0
           
            # short_share buy should occur if position dataframe  
            # contains a -1 and there are no short_shares held
            if (position_df.iloc[position_row, position_column] == -1
            and short_shares == 0):
                short_shares = 100.00 / df_2017.loc[position_row, 'close']   
                short_price = df_2017.loc[position_row, 'close']
                short_worth = 0
            
            # Sell section
            # long_shares sell should occur if position dataframe  
            # contains a -1 and there are long_shares held
            if (position_df.iloc[position_row, position_column] == -1
            and long_shares != 0): 
                long_worth = ((long_shares 
                              * df_2017.loc[position_row, 'close'])
                              - long_price)
                long_shares = 0
                long_price = 0
                
            # short_shares sell should occur if position dataframe  
            # contains a 1 and there are short_shares held
            if (position_df.iloc[position_row, position_column] == 1
            and short_shares != 0): 
                short_worth = ((short_shares 
                              * df_2017.loc[position_row, 'close'])
                              - short_price)
                short_shares = 0
                short_price = 0
                
            # On each loop iteration record the current position long/short
            # stocks held and the profits made from their sales
            stocks_df.at[position_row, 'long_shares']  = long_shares
            stocks_df.at[position_row, 'long_worth']   = long_worth
            stocks_df.at[position_row, 'long_price']   = long_price
            stocks_df.at[position_row, 'short_shares'] = short_shares
            stocks_df.at[position_row, 'short_worth']  = short_worth
            stocks_df.at[position_row, 'short_price']  = short_price
            
            # Manual increments
            #position_column += 1
            position_row += 1


print(sum(stocks_df.iloc[:, 2]) / 251
sum(stocks_df.iloc[:, 3]) / 251


        
except Exception as e:
    print(e)
    sys.exit('Failed to evaluate stock trades for 2017')




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

