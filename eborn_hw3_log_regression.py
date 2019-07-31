# -*- coding: utf-8 -*-
"""
Eric Born
Class: CS677 - Summer 2
Date: 30 July 2019
Homework week 3 - stock logistic regression
Create a logistic regression model based upon the mean and standard
deviation measures of the weekly stock returns
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
#import scipy
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix


# setup input directory and filename
ticker = 'BSX-labeled'
input_dir = r'C:\Users\TomBrody\Desktop\School\677\wk3\logistic regression'
ticker_file = os.path.join(input_dir, ticker + '.csv')

# read csv file into dataframe
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
        'volume', 'adj_close', 'returns', 'short_ma', 'long_ma', 'label']

# Create dataframe
bsx_df = pd.DataFrame([sub.split(',') for sub in lines], columns = cols)

# Drop the first row which contains the header since the column names are
# set during the dataframe construction
bsx_df = bsx_df.drop([0], axis = 0)

# Create class column where red = 0 and green = 1
try:
    bsx_df['class'] = bsx_df['label'].apply(lambda x: 1 if x =='green' else 0)
    
except Exception as e:
    print(e)
    print('failed to add class column to bsx_df') 

# explicitly set column types for ease in calculating
bsx_df.td_year = bsx_df.td_year.astype(int)
bsx_df.returns = bsx_df.returns.astype(float)

# create separate dataframes for 2017 and 2018 data
df_2017 = bsx_df.loc[bsx_df['td_year']==2017]
df_2018 = bsx_df.loc[bsx_df['td_year']==2018]

# reset indexes
#df_2017 = df_2017.reset_index(level=0, drop=True)
#df_2018 = df_2018.reset_index(level=0, drop=True)  

# Create reduced dataframe only containing week number, mu, sig and label
df_2017_reduced = pd.DataFrame( {'week nbr' : range(1, 53),
                'mu'    : df_2017.groupby('td_week_number')['returns'].mean(),
                'sig'   : df_2017.groupby('td_week_number')['returns'].std(),
                'label' : df_2017.groupby('td_week_number')['class'].first()})

# Create reduced dataframe only containing week number, mu, sig and label
df_2018_reduced = pd.DataFrame( {'week nbr' : range(0, 53),
                'mu'    : df_2018.groupby('td_week_number')['returns'].mean(),
                'sig'   : df_2018.groupby('td_week_number')['returns'].std(),
                'label' : df_2018.groupby('td_week_number')['class'].first()})

# reset indexes
df_2017_reduced = df_2017_reduced.reset_index(level=0, drop=True)
df_2018_reduced = df_2018_reduced.reset_index(level=0, drop=True)      
    
    
# Replacing nan in week 52 sigma column with a zero due to 
# there being only 1 trading day that week.
df_2018_reduced = df_2018_reduced.fillna(0)

# remove index name labels from dataframes
#del df_2017_reduced.index.name
#del df_2018_reduced.index.name

# Define features and class labels
features = ['mu', 'sig']
class_labels = ['green', 'red']

# create x training and test sets from 2017/2018 features values
x_train = df_2017_reduced[features].values
x_test = df_2018_reduced[features].values

# create y training and test sets from 2017/2018 label values
y_train = df_2017_reduced['label'].values
y_test = df_2018_reduced['label'].values

# Setup scalers as without the classifier was just guessing all 0's
# Scaler for training data
scaler = StandardScaler()
scaler.fit(x_train)
x_2017_train = scaler.transform(x_train)

# scaler for test data
scaler.fit(x_test)
x_2018_test = scaler.transform(x_test)

# Create a logistic classifier
# set solver to avoid the warning
log_reg_classifier = LogisticRegression(solver = 'liblinear')

# Train the classifier on 2017 data
log_reg_classifier.fit(x_2017_train, y_train)

# Predict using 2018 feature data
prediction = log_reg_classifier.predict(x_2018_test)

# print coefficient and intercept
print(log_reg_classifier.coef_)
print(log_reg_classifier.intercept_)

# print coefficients with feature names
coef = log_reg_classifier.coef_
for p,c in zip(features,list(coef[0])):
    print(p + '\t' + str(c))

# 1)
# The equation for logistic regression found in year 1 data
#y = 0.1621 + 2.0277*mu - 0.0168*sig

# 2)
# Check the predicitons accuracy
# 79.245% accuracy
accuracy = np.mean(prediction == y_test)
print(accuracy)

# 3)
# Output the confusion matrix
cm = confusion_matrix(y_test, prediction)
print(cm)

# Create confusion matrix heatmap
# setup class names and tick marks
class_names=[0,1]
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

# create heatmap and labels
sns.heatmap(pd.DataFrame(cm), annot=True, cmap="BrBG" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

# 4)	
#TPR = 20/29 = 0.69 = 69%
#TNR = 22/24 = 0.917 = 91.7%

# 5)
# Buy and hold
# Initialize wallet and shares to track current money and number of shares.
wallet = 100.00
shares = 0
worth = 0

# stores adj_close values for the last day of each trading week
adj_close = df_2018.groupby('td_week_number')['adj_close'].last()

# stores open price for the first day of each trading week
open_price = df_2018.groupby('td_week_number')['open'].first()



try:
    for i in range(0, len(prediction)):
        # Sell should occur on the last day of a green week at 
        # the adjusted_close price. Since i is tracking the current
        # trading week we need to minus 1 to get the adjusted close price
        # from the previous trading week
        if prediction[i] == 0 and shares > 0:
            wallet = round(shares * adj_close[i - 1], 2)
            shares = 0
            
        # Buy should occur on the first day of a green week at the open price
        if prediction[i] == 1 and shares == 0: 
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
#Worth $ 141.7
#Selling on the final day would result in $ 141.7 a profit of $ 41.7
print('\n2018 buy and hold:','\nCurrently own', shares, 'shares',
      '\nWorth','$',round(worth, 2))
print('Selling on the final day would result in $', worth, 'a profit of $', profit)


