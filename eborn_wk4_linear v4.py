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

# list that contains the window sizes
window_size = []

# 1)
# This section of code calculates the regression for each day incrementing
# through a window size from 5 to 30 days.
# If the predicted close price for w+1 is greater than the close price for w,
# a 1 is put into the position column in the df_2017 dataframe.

# If the predicted close price for w+1 is less than the close price for w
# a -1 is put into the position column in the df_2017 dataframe.

# If the predicted close price for w+1 is equal to the close price for w
# a 0 is put into the position column in the df_2017 dataframe.


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
    
    # window size list populated with size increments
    window_size.append(window)
    
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
        if float(pred) > df_2017.loc[window_end, 'close']:
            df_2017.loc[window_end, 'position'] = 1
        elif float(pred) == df_2017.loc[window_end, 'close']:
            df_2017.loc[window_end, 'position'] = 0
        else:
            df_2017.loc[window_end, 'position'] = -1
        window_start += 1
        window_end += 1
    
    # writes the position column to a the position dataframe after each
    # window iteration
    position_df[str(window)] = df_2017.loc[:, 'position']
    
# Initialize variables and trade_data_df to their starting values
# before they are utilized in the loop to build out the trade_data df
long_shares = 0
long_worth = 0
long_price = 0
name_increment = 5
trade_data_df = pd.DataFrame()

# Manual variable setters for testing
#position_df.iloc[:, 0]
#position_column = 0
#position_row = 4

# for loop that evaluates the dataset deciding when to buy/sell based
# upon the prediction labels. 0 is a bad week, 1 is a good week
try:
    for position_column in range(0, len(position_df.iloc[0, :])):
        # used to increment the column name to represend the window size
        long_price_name  = 'long_price'  + str(name_increment)
        long_shares_name = 'long_shares' + str(name_increment)
        long_worth_name  = 'long_worth'  + str(name_increment)

        for position_row in range(0, len(position_df)):
            # Buy section
            # long_shares buy should occur if position dataframe  
            # contains a 1 and there are no long_shares held
            if (position_df.iloc[position_row, position_column] == 1 
            and long_shares == 0): 
                long_shares = 100.00 / df_2017.loc[position_row, 'close']           
                long_price = df_2017.loc[position_row, 'close']
                trade_data_df.at[position_row, long_price_name] = long_price
                #trade_data_df.at[position_row, long_worth_name] = ((long_shares 
                #              * df_2017.loc[position_row, 'close'])
                #              - long_price * long_shares)
            
            # Sell section
            # long_shares sell should occur if position dataframe  
            # contains a -1 and there are long_shares held
            if (position_df.iloc[position_row, position_column] == -1
            and long_shares != 0): 
                long_worth = ((long_shares 
                              * df_2017.loc[position_row, 'close'])
                              - long_price * long_shares)
                trade_data_df.at[position_row, long_worth_name] = (
                                                          round(long_worth, 2))
                trade_data_df.at[position_row, long_price_name] = (
                                            df_2017.loc[position_row, 'close'])
                long_shares = 0
                long_price = 0
                long_worth = 0
                  
            # On each loop iteration record the current long shares held
            trade_data_df.at[position_row, long_shares_name]  = long_shares
       
            # Manual increments for testing
            #position_column += 1
            #position_row += 1

        # increments the name_increment to represent the window size
        name_increment += 1
            
except Exception as e:
    print(e)
    sys.exit('failed to build trading data for trade_data')            

# NaN are excluded when using the mean function so I decided to leave them in
# Replace all NaN with 0's
#trade_data_df = trade_data_df.fillna(0)            

# export trade data to CSV
#try:  
#    trade_data_df.to_csv(r'C:\Users\TomBrody\Desktop\School\677\wk4\trade_data.csv', index = False)
#
#except Exception as e:
#    print(e)
#    sys.exit('failed to export trade_data_df to csv')     

# sample data selections
#trade_data_df.iloc[0:10, 0:3]
#trade_data_df.iloc[0:10, 77]
#trade_data_df.iloc[0:3, 2]
#trade_data_df.iloc[:, column]

# creates a list containing the column names from the trade_data_df
name_list = []
for column in range(2, len(trade_data_df.iloc[0, :]), 3):
    name_list.append(trade_data_df.iloc[:, column].name)

# create a dataframe to store the daily profits made from selling stocks
summary_df = trade_data_df[name_list].copy()

# Sum of profits by window size
summary_df.sum(axis = 0)

# mean of profits by window size
summary_df.mean(axis = 0)

# creates a barplot of the window size vs the average return in dollars
sns.barplot(window_size, summary_df.sum(axis = 0), palette = 'Blues_d')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('window Size vs. Total Return')
plt.xlabel('window Size')
plt.ylabel('Total Return in Dollars')

# creates a barplot of the window size vs the average return in dollars
sns.barplot(window_size, summary_df.mean(axis = 0), palette = 'Blues_d')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Window Size vs. Average Return')
plt.xlabel('Window Size')
plt.ylabel('Avg Return in Dollars')






 
plt.line(window_size, summary_df.mean(axis = 0), color='red')


plt.show()






    
    summary_df.at[trade_data_df.iloc[:, column], trade_data_df.iloc[:, column].name] = 
    
    
    
    
    print(round(trade_data_df.iloc[:, column], 2))
    print(round(np.mean(trade_data_df.iloc[, column]), 2))
    #print(round(sum(trade_data_df.iloc[:, column]), 2))


'''
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
'''
