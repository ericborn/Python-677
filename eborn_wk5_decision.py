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
from sklearn import tree
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, recall_score

# setup input directory and filename
ticker = 'BSX-labeled'
input_dir = r'C:\Users\TomBrody\Desktop\School\677\wk5\decision'
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

# Create a decisions tree classifier
tree_clf = tree.DecisionTreeClassifier(criterion = 'entropy')

# Train the classifier on 2017 data
tree_clf = tree_clf.fit(x_train_2017, y_train_2017)

# Predict using 2018 feature data
prediction = tree_clf.predict(x_test_2018)

# calculate error rate
error_rate = 100-(round(np.mean(prediction != y_test_2018) * 100, 2))

# 1)
# Print error rate
print('The decision tree classifier has an accuracy of', error_rate,'%')

# 2)









