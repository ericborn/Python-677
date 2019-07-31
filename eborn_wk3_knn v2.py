# -*- coding: utf-8 -*-
"""
Eric Born
Class: CS677 - Summer 2
Date: 30 July 2019
Homework week 3 - stock logistic regression
Create a KNN model based upon the mean and standard
deviation measures of the weekly stock returns
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# setup input directory and filename
ticker = 'BSX-labeled'
input_dir = r'C:\Users\TomBrody\Desktop\School\677\wk3\knn'
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

# remove index name labels from dataframes
del df_2017_reduced.index.name
del df_2018_reduced.index.name

# Define features labels
features = ['mu', 'sig']

# create x dataset from 2017 features values
X = df_2017_reduced[features].values

# create y datasets from 2017 label values
Y = df_2017_reduced['label'].values

# Setup scalers X dataset
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

# divide X and Y into test/train 50/50
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.5, 
                                                    random_state = 3)

# Create empty lists to store the models error rate and 
# accuracy across various K's
error_rate = []
accuracy = []
k_value = []

# Test the model using 2018 data with k neighbors set to 3,5,7,9 and 11
try:
    for k in range (3, 13, 2):
        # Create the classifier with neighbors set to k from the loop
        knn_classifier = KNeighborsClassifier(n_neighbors = k)
       
        # Train the classifier
        knn_classifier.fit(X_train, Y_train)
        
        # Perform predictions
        pred_k = knn_classifier.predict(X_test)
        
        # Store error rate and accuracy for particular K value
        k_value.append(k)
        error_rate.append(round(np.mean(pred_k != Y_test) * 100, 2))
        accuracy.append(round(sum(pred_k == Y_test) / len(pred_k) * 100, 2))
except Exception as e:
    print(e)
    print('failed to build the KNN classifier.')

for i in range (0,5):
    print('For K =', k_value[i], 'the accuracy is:', accuracy[i])
    
# create a plot to display the accuracy of the model across K
fig = plt.figure(figsize=(10, 4))
ax = plt.gca()
plt.plot(range(3, 13, 2), accuracy, color ='blue',
         marker = 'o', markerfacecolor = 'black', markersize = 10)
plt.title('Accuracy vs. k for stock labels')
plt.xlabel('Number of neighbors: k')
plt.ylabel('Accuracy')


###########
# create x and y test sets for 2018
x_test = df_2018_reduced[features].values
y_test = df_2018_reduced['label'].values

# scaler for 2018 test data
scaler.fit(x_test)
x_2018_test = scaler.transform(x_test)
