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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, recall_score

# setup input directory and filename
ticker = 'BSX-labeled'
input_dir = r'C:\Users\TomBrody\Desktop\School\677\wk3\logistic regression'
ticker_file = os.path.join(input_dir, ticker + '.csv')

# read csv file into dataframe
try:
    df = pd.read_csv(ticker_file)
    print('opened file for ticker: ', ticker,'\n')

except Exception as e:
    print(e)
    print('failed to read stock data for ticker: ', ticker)

# Create class column where red = 0 and green = 1
df['class'] = df['label'].apply(lambda x: 1 if x =='green' else 0)

# Create separate dataframes for 2017 and 2018 data
# 2017 will be used as training, 2018 as testing for the model
df_2017 = df.loc[df['td_year']==2017]
df_2018 = df.loc[df['td_year']==2018]

# Reset indexes
df_2017 = df_2017.reset_index(level=0, drop=True)
df_2018 = df_2018.reset_index(level=0, drop=True)  

# Create reduced dataframe only containing week number, mu, sig and label
df_2017_reduced = pd.DataFrame( {'week nbr' : range(1, 53),
                'mu'    : df_2017.groupby('td_week_number')['return'].mean(),
                'sig'   : df_2017.groupby('td_week_number')['return'].std(),
                'label' : df_2017.groupby('td_week_number')['class'].first()})

# Create reduced dataframe only containing week number, mu, sig and label
df_2018_reduced = pd.DataFrame( {'week nbr' : range(0, 53),
                'mu'    : df_2018.groupby('td_week_number')['return'].mean(),
                'sig'   : df_2018.groupby('td_week_number')['return'].std(),
                'label' : df_2018.groupby('td_week_number')['class'].first()})

# Replacing nan in week 52 sigma column with a zero due to 
# there being only 1 trading day that week.
df_2018_reduced = df_2018_reduced.fillna(0)

# Remove index name labels from dataframes.
del df_2017_reduced.index.name
del df_2018_reduced.index.name

# Define features labels.
features = ['mu', 'sig']

# Create x training and test sets from 2017/2018 features values.
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

# Scaler for test data
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
# print(log_reg_classifier.coef_)
# print(log_reg_classifier.intercept_)

print('output the coefficients with feature names')
coef = log_reg_classifier.coef_
for p,c in zip(features,list(coef[0])):
    print(p + '\t' + str(c))

# 1)
# what is the equation for logistic regression that your 
# classifier found from year 1 data?
print('\nThe equation for logistic regression found in year 1 data is:')
print('y = 0.1621 + 2.0277*mu - 0.0168*sig')

# 2)
# what is the accuracy for year 2?
# 79.245% accuracy
accuracy = np.mean(prediction == y_test)
print('\nThe accuracy for year 2 is:')
print(round(accuracy * 100, 2), '%')

# 3)
# Output the confusion matrix
cm = confusion_matrix(y_test, prediction)
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
sns.heatmap(pd.DataFrame(cm), annot=True, cmap="BrBG" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

# 4)
# what is true positive rate (sensitivity or recall) and true
# negative rate (specificity) for year 2?
print('The recall is: 22/24 =', 
      round(recall_score(y_test, prediction) * 100, 2),'%')
print('The specificity is: 20/29 = 68.97%')	

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
#Worth $ 141.70
#Selling on the final day would result in $ 141.7 a profit of $ 41.7
print('\n2018 buy and hold:','\nCurrently own', shares, 'shares',
      '\nWorth','$',"%.2f"%round(worth, 2))
print('Selling on the final day would result in $',"%.2f"%worth, 'a profit of $', "%.2f"%profit)


