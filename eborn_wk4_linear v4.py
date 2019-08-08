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
#from scipy import stats

'''
Function created by professor Pinsky
takes x and y as inputs and calculates the size of the array,
the sum of squares, slope and intercept.
'''
def estimate_coef (x, y):
    n = np.size(x)
    mu_x, mu_y = np.mean(x), np.mean(y)
    SS_xy = np.sum(y * x) - n * mu_y * mu_x
    SS_xx = np.sum(x * x) - n * mu_x * mu_x
    slope = SS_xy / SS_xx
    intercept = mu_y - slope * mu_x
    return (slope, intercept)

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
df_2017 = df.loc[df['td_year']==2017]
df_2018 = df.loc[df['td_year']==2018]

# Reset indexes
df_2017 = df_2017.reset_index(level=0, drop=True)
df_2018 = df_2018.reset_index(level=0, drop=True)  

# Creates a linear regression object
lm = LinearRegression()

# stores the position 0, 1, -1 for each window size
position_2017_df  = pd.DataFrame()

# list that will be populated from the below for loop
# to contains the window sizes
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
try:
    for window in range(5,31):
        # Create position column to indicate whether its a buy or sell day.
        # column is reset to all 0's at the start of each loop iteration
        df_2017['position'] = 0
        df_2017['prediction'] = 0
        
        # window size list populated with size increments
        window_size.append(window)
        
        # set window_end equal to window - 1 due to zero index
        window_start = 0
        window_end = window - 1
        
        # loop that handles gathering the adj_close and close price 
        # for the appropriate window size
        for rows in range(0, len(df_2017)):
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
            
            # store the predicted value in the 2018 dataframe
            df_2017.loc[window_end + 1, 'prediction'] = float(pred) 
    
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
        position_2017_df[str(window)] = df_2017.loc[:, 'position']

except Exception as e:
    print(e)
    sys.exit('Failed to perform predictions on 2017 data')
    
# Initialize variables and trade_data_df to their starting values
# before they are utilized in the loop to build out the trade_data df
long_shares = 0
long_worth = 0
long_price = 0
name_increment = 5
trade_data_2017_df = pd.DataFrame()

# Manual variable setters for testing
#position_df.iloc[:, 0]
#position_column = 0
#position_row = 4

# for loop that evaluates the dataset deciding when to buy/sell based
# upon the prediction labels. 0 is a bad week, 1 is a good week
try:
    for position_column in range(0, len(position_2017_df.iloc[0, :])):
        # used to increment the column name to represend the window size
        long_price_name  = 'long_price'  + str(name_increment)
        long_shares_name = 'long_shares' + str(name_increment)
        long_worth_name  = 'long_worth'  + str(name_increment)

        for position_row in range(0, len(position_2017_df)):
            # Buy section
            # long_shares buy should occur if position dataframe  
            # contains a 1 and there are no long_shares held
            if (position_2017_df.iloc[position_row, position_column] == 1 
            and long_shares == 0): 
                long_shares = 100.00 / df_2017.loc[position_row, 'close']           
                long_price = df_2017.loc[position_row, 'close']
                trade_data_2017_df.at[position_row, long_price_name] = long_price
                #trade_data_df.at[position_row, long_worth_name] = ((long_shares 
                #              * df_2017.loc[position_row, 'close'])
                #              - long_price * long_shares)
            
            # Sell section
            # long_shares sell should occur if position dataframe  
            # contains a -1 and there are long_shares held
            if (position_2017_df.iloc[position_row, position_column] == -1
            and long_shares != 0): 
                long_worth = ((long_shares 
                              * df_2017.loc[position_row, 'close'])
                              - long_price * long_shares)
                trade_data_2017_df.at[position_row, long_worth_name] = (
                                                          round(long_worth, 2))
                trade_data_2017_df.at[position_row, long_price_name] = (
                                            df_2017.loc[position_row, 'close'])
                long_shares = 0
                long_price = 0
                long_worth = 0
                  
            # On each loop iteration record the current long shares held
            trade_data_2017_df.at[position_row, long_shares_name]  = long_shares
       
            # Manual increments for testing
            #position_column += 1
            #position_row += 1

        # increments the name_increment to represent the window size
        name_increment += 1
            
except Exception as e:
    print(e)
    sys.exit('Failed to build trading data for trade_data_df')            

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
for column in range(2, len(trade_data_2017_df.iloc[0, :]), 3):
    name_list.append(trade_data_2017_df.iloc[:, column].name)

# create a dataframe to store the daily profits made from selling stocks
summary_2017_df = trade_data_2017_df[name_list].copy()

# Sum of profits by window size
profit_2017 = summary_2017_df.sum()
print(profit_2017)

# creates a barplot of the window size vs the sum of profits in dollars
sns.barplot(window_size, summary_2017_df.sum(), palette = 'Blues_d')
plt.tight_layout()
plt.title('Window Size vs. Total Return')
plt.xlabel('Window Size')
plt.ylabel('Total Return in Dollars')
plt.show()

# mean of profits by window size
mean_2017 = summary_2017_df.mean()
print(mean_2017)

# creates a barplot of the window size vs the average return in dollars
sns.lineplot(window_size, summary_2017_df.mean(), color = 'navy')
plt.tight_layout()
plt.title('Window Size vs. Average Return')
plt.xlabel('Window Size')
plt.ylabel('Avg Return in Dollars')
plt.show()

# creates a barplot of the window size vs the average return in dollars
sns.lineplot(window_size, summary_2017_df.sum(), color = 'navy')
plt.tight_layout()
plt.title('Window Size vs. Total Return')
plt.xlabel('Window Size')
plt.ylabel('Total Return in Dollars')
plt.show()

# Review the number of trades by window size
print(summary_2017_df.count())

# creates a barplot of the number of trades by window size
sns.barplot(window_size, summary_2017_df.count(), palette = 'Blues_d')
plt.tight_layout()
plt.title('Window Size vs. Number of Trades')
plt.xlabel('Window Size')
plt.ylabel('Total Number of Trades')
plt.show()


###### Part 2
# Using a fixed window size of 5 which was determined in part 1 above
# to analyze and predict stock prices in data from 2018.
# Unfortunately the two measures being used, closing and adjusted 
# closing price are exactly the same every day in 2018
# so the results are a bit uninteresting

# Setup column in the 2018 dataframe to track the buy/sell position,
# the predicted value and initial start and end window positions
df_2018['position'] = 0
df_2018['prediction'] = 0
window_start = 0
window_end = 4

# stores the position 0, 1, -1 for each window size
position_2018_df  = pd.DataFrame()
try:    
    # loop that handles gathering the adj_close and close price 
    # for the 2018 dataframe
    for rows in range(0, len(df_2018)):
        adj_close = np.array(df_2018.loc[window_start:window_end,
                                 'adj_close']).reshape(-1, 1)
        close = np.array(df_2018.loc[window_start:window_end,
                                 'close']).reshape(-1, 1)
        # fits the model using adjusted close and close stock prices
        lm.fit(adj_close, close)
        
        # Breaks on the last row since it cannot predict w + 1 if 
        # there is no data for the next day, else it creates
        # a prediction.
        if window_end == len(df_2018) - 1:
            break
        else:
            pred = lm.predict(np.array(df_2018.loc[window_end + 1, 
                                      'close']).reshape(-1, 1))
        
        # store the predicted value in the 2018 dataframe
        df_2018.loc[window_end + 1, 'prediction'] = float(pred) 
        
        # updates the position column with a 1 when prediciton for tomorrows
        # close price (w + 1) is greater than the close price of w.
        # Else it marks it with a -1 to indicate a lower price.
        if float(pred) > df_2018.loc[window_end, 'close']:
            df_2018.loc[window_end, 'position'] = 1
        elif float(pred) == df_2018.loc[window_end, 'close']:
            df_2018.loc[window_end, 'position'] = 0
        else:
            df_2018.loc[window_end, 'position'] = -1
        window_start += 1
        window_end += 1

        # writes the position column to a the position dataframe after each
        # window iteration
        position_2018_df[str(window)] = df_2018.loc[:, 'position']

except Exception as e:
    print(e)
    sys.exit('Failed to build prediction data for 2018')       

# Initialize variables and trade_data_df to their starting values
# before they are utilized in the loop to build out the trade_data df
long_shares = 0
long_worth = 0
long_price = 0
name_increment = 5
trade_data_2018_df = pd.DataFrame()

# Manual variable setters for testing
#position_2018_df.iloc[:, 0]
#position_column = 0
#position_row = 4

# for loop that evaluates the dataset deciding when to buy/sell based
# upon the prediction labels. 0 is a bad week, 1 is a good week
try:
    for position_column in range(0, len(position_2018_df.iloc[0, :])):
        # used to increment the column name to represend the window size
        long_price_name  = 'long_price'  + str(name_increment)
        long_shares_name = 'long_shares' + str(name_increment)
        long_worth_name  = 'long_worth'  + str(name_increment)

        for position_row in range(0, len(position_2018_df)):
            # Buy section
            # long_shares buy should occur if position dataframe  
            # contains a 1 and there are no long_shares held
            if (position_2018_df.iloc[position_row, position_column] == 1 
            and long_shares == 0): 
                long_shares = 100.00 / df_2018.loc[position_row, 'close']           
                long_price = df_2018.loc[position_row, 'close']
                trade_data_2018_df.at[position_row, long_price_name] = long_price
                #trade_data_df.at[position_row, long_worth_name] = ((long_shares 
                #              * df_2017.loc[position_row, 'close'])
                #              - long_price * long_shares)
            
            # Sell section
            # long_shares sell should occur if position dataframe  
            # contains a -1 and there are long_shares held
            if (position_2018_df.iloc[position_row, position_column] == -1
            and long_shares != 0): 
                long_worth = ((long_shares 
                              * df_2018.loc[position_row, 'close'])
                              - long_price * long_shares)
                trade_data_2018_df.at[position_row, long_worth_name] = (
                                                          round(long_worth, 2))
                trade_data_2018_df.at[position_row, long_price_name] = (
                                            df_2018.loc[position_row, 'close'])
                long_shares = 0
                long_price = 0
                long_worth = 0
                  
            # On each loop iteration record the current long shares held
            trade_data_2018_df.at[position_row, long_shares_name]  = long_shares
       
            # Manual increments for testing
            #position_column += 1
            #position_row += 1

        # increments the name_increment to represent the window size
        name_increment += 1
            
except Exception as e:
    print(e)
    sys.exit('Failed to build trading data for trade_data_df')            

# Finish data building
#########
# Start analysis/presenation
    
# create a dataframe to store the daily profits made from selling stocks
summary_2018_df = trade_data_2018_df['long_worth5'].copy()

# Creating an estimated coefficient between the measures
# the slope is a perfect 1 and the intercept is at 0
# dataframes start at position 5 since the first 0-4 were not predicted
# do to window size starting at 5
coefficient = estimate_coef(df_2018.loc[5:,'adj_close'], df_2018.loc[5:,'prediction'])
print(coefficient)

# Generate a plot of the actual vs predicted values
sns.scatterplot(df_2018.loc[5:,'adj_close'], df_2018.loc[5:,'prediction'], color='navy')
sns.lineplot(range(25, 40), range(25, 40), color = 'red')
plt.title('Actual Close vs. Predicted Close')
plt.xlabel('Actual Close')
plt.ylabel('Predicted Close')
plt.show()

# average R2 value
# 0.5
print(np.mean(coefficient))

# long days
# 131
long_days = df_2018[df_2018['position'] > 0].count()['position']
print(long_days)

# short days
# 111
short_days = df_2018[df_2018['position'] < 0].count()['position']
print(short_days)

# Plot long vs short day totals
plot_data = pd.DataFrame({'Type': ['Long', 'Short'], 
                          'Days': [int(long_days), int(short_days)]})
sns.barplot(x = 'Type', y = 'Days', data = plot_data) 
plt.title('Long Days vs. Short Days')
plt.xlabel('Type')
plt.ylabel('Total Days')
plt.show()

# Sum of profits for 2018
profit_2018 = summary_2018_df.sum()
print(profit_2018)

# Plot long vs short day totals
profit_data = pd.DataFrame({'Year': ['2017', '2018'], 
                            'Profit': [int(profit_2017[0]), int(profit_2018)]})
sns.barplot(x = 'Year', y = 'Profit', data = profit_data) 
plt.title('Total Profit 2017 vs. 2018')
plt.xlabel('Year')
plt.ylabel('Total Profit in Dollars')
plt.show()

# mean of profits by window size
print(summary_2018_df.mean())

# Review the number of trades by window size
print(summary_2018_df.count())


