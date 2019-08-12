# -*- coding: utf-8 -*-
"""
Eric Born
Class: CS677 - Summer 2
Date: 12 Aug 2019
Homework week 5 - stock decision tree
Create a decision tree model based upon the mean and standard
deviation measures of the weekly stock returns
"""

import os
from sys import exit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, recall_score

# setup input directory and filename
ticker = 'BSX-labeled'
input_dir = r'C:\Users\TomBrody\Desktop\School\677\wk5'
ticker_file = os.path.join(input_dir, ticker + '.csv')

# read csv file into dataframe
try:
    bsx_df = pd.read_csv(ticker_file)
    print('opened file for ticker: ', ticker,'\n')

except Exception as e:
    print(e)
    exit('failed to read stock data for ticker: '+ str(ticker))

# Create class column where red = 0 and green = 1
bsx_df['class'] = bsx_df['label'].apply(lambda x: 1 if x =='green' else 0)

# Create separate dataframes for 2017 and 2018 data
# 2017 will be used as training, 2018 as testing for the model
bsx_df_2017 = bsx_df.loc[bsx_df['td_year']==2017]
bsx_df_2018 = bsx_df.loc[bsx_df['td_year']==2018]

# Reset indexes
bsx_df_2017 = bsx_df_2017.reset_index(level=0, drop=True)
bsx_df_2018 = bsx_df_2018.reset_index(level=0, drop=True)  

# Create reduced dataframe only containing week number, mu, sig and label
bsx_2017_reduced = pd.DataFrame( {'week nbr' : range(1, 53),
            'mu'    : bsx_df_2017.groupby('td_week_number')['return'].mean(),
            'sig'   : bsx_df_2017.groupby('td_week_number')['return'].std(),
            'label' : bsx_df_2017.groupby('td_week_number')['class'].first()})

# Create reduced dataframe only containing week number, mu, sig and label
bsx_2018_reduced = pd.DataFrame( {'week nbr' : range(0, 53),
            'mu'    : bsx_df_2018.groupby('td_week_number')['return'].mean(),
            'sig'   : bsx_df_2018.groupby('td_week_number')['return'].std(),
            'label' : bsx_df_2018.groupby('td_week_number')['class'].first()})

# Replacing nan in week 52 sigma column with a zero due to 
# there being only 1 trading day that week.
bsx_2018_reduced = bsx_2018_reduced.fillna(0)

# Remove index name labels from dataframes.
del bsx_2017_reduced.index.name
del bsx_2018_reduced.index.name

# Define features labels.
features = ['mu', 'sig']

# Create x and y training and test sets from 2017 data
x_train_2017 = bsx_2017_reduced[features].values
y_train_2017 = bsx_2017_reduced['label'].values

# Create x and y training and test sets from 2018 data
x_test_2018 = bsx_2018_reduced[features].values
y_test_2018 = bsx_2018_reduced['label'].values

# Create a naive bayes classifier
gnb = GaussianNB()

# Train the classifier on 2017 data
gnb_fit = gnb.fit(x_train_2017, y_train_2017)

# Predict using 2018 feature data
prediction = gnb_fit.predict(x_test_2018)

# calculate error rate
accuracy_rate = 100-(round(np.mean(prediction != y_test_2018) * 100, 2))

# 1)
# Print error rate
print('The decision tree classifier has an accuracy of', accuracy_rate,'%')

# 2)
# Output the confusion matrix
cm = confusion_matrix(y_test_2018, prediction)
print('\nConfusion matrix for year 2 predictions:')
print(cm, '\n')

# Create confusion matrix heatmap
# setup class names and tick marks
class_names=[0,1]
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

# Create heatmap and labels
sns.heatmap(pd.DataFrame(cm), annot=True, cmap="summer" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

# 3)
# store confusion matrix figures
tn, fp, fn, tp = confusion_matrix(y_test_2018, prediction).ravel()

# TPR/TNR rates
print('The TPR is:', str(tp) + '/' + str(tp + fn) + ',',
      round(recall_score(y_test_2018, prediction) * 100, 2),'%')
print('The TNR is:', str(tn) + '/' + str(tn + fp) + ',',
    round(tn / (tn + fp) * 100, 2),'%')

# 4)
# Implemented trading strategy based upon label predicitons vs
# buy and hold strategy

# Initialize wallet and shares to track current money and number of shares.
wallet = 100.00
shares = 0
worth = 0

# stores adj_close values for the last day of each trading week
adj_close = bsx_df_2018.groupby('td_week_number')['adj_close'].last()

# stores open price for the first day of each trading week
open_price = bsx_df_2018.groupby('td_week_number')['open'].first()

# for loop that evaluates the dataset deciding when to buy/sell based
# upon the prediction labels. 0 is a bad week, 1 is a good week
try:
    for day in range(0, len(prediction)):
        # Sell should occur on the last day of a green week at 
        # the adjusted_close price. Since i is tracking the current
        # trading week we need to minus 1 to get the adjusted close price
        # from the previous trading week
        if prediction[day] == 0 and shares > 0:
            wallet = round(shares * adj_close[day - 1], 2)
            shares = 0
            
        # Buy should occur on the first day of a green week at the open price
        if prediction[day] == 1 and shares == 0: 
            shares = wallet / open_price[day]
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
