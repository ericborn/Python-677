# -*- coding: utf-8 -*-
"""
Eric Born
Class: CS677 - Summer 2
Date: 14 Aug 2019
Homework week 6 - stock SVM
Create a SVM model based upon the mean and standard
deviation measures of the weekly stock returns
"""

import os
from sys import exit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, recall_score
from sklearn.preprocessing import StandardScaler

# setup input directory and filename
ticker = 'BSX-labeled'
input_dir = r'C:\Users\TomBrody\Desktop\School\677\wk6'
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

# Define features labels.
features = ['mu', 'sig']

# Create reduced dataframe only containing week number, mu, sig and label
bsx_reduced = pd.DataFrame( {'week nbr' : range(0, 53),
            'mu'    : bsx_df.groupby('td_week_number')['return'].mean(),
            'sig'   : bsx_df.groupby('td_week_number')['return'].std(),
            'label' : bsx_df.groupby('td_week_number')['class'].first()})

# Remove index name labels from dataframes
del bsx_reduced.index.name
    
# Create x and y training and test sets from 2017 data
x_values = bsx_reduced[features].values
y_values = bsx_reduced['label'].values


# create the k-means classifier with 3 clusters
kmeans_classifier = KMeans(n_clusters = 3)
y_means = kmeans_classifier.fit_predict(x_values)
centroids = kmeans_classifier.cluster_centers_

# create empty list to store inertia values
inertia_list = []

# for loop that iterates k 1 through 8
for k in range(1, 9):
    kmeans_classifier = KMeans(n_clusters = k)
    y_means = kmeans_classifier.fit_predict(x_values)
    inertia = kmeans_classifier.inertia_
    inertia_list.append(inertia)

# Generate a plot based upon the iterations of k
fig,ax = plt.subplots(1,figsize =(7,5))
plt.plot(range(1, 9), inertia_list, marker='o',
        color='green')
plt.legend()
plt.xlabel('number of clusters: k')
plt.ylabel('inertia')
plt.tight_layout()
plt.show()

# 2)
colmap = {0: 'red', 1: 'green'}

# create the k-means classifier with 2 clusters
kmeans_classifier = KMeans(n_clusters = 2)
y_means = kmeans_classifier.fit_predict(x_values)
centroids = kmeans_classifier.cluster_centers_


fig, ax = plt.subplots(1,figsize =(7,5))
plt.scatter(x_values[y_means == 0, 0], x_values[y_means == 0, 1],
                s = 75, c ='red', label = 'red')
plt.scatter(x_values[y_means == 1, 0], x_values[y_means == 1, 1],
                s = 75, c = 'green', label = 'green')
plt.scatter(centroids[:, 0], centroids[:,1] ,
                s = 200 , c = 'black', label = 'Centroids')


# 1)
# Print accuracy rate
print('The linear SVM classifier has an accuracy of', accuracy_rate_linear,'%')

# 2)
# Output the confusion matrix
cm = confusion_matrix(y_test_2018, prediction_linear)
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
tn, fp, fn, tp = confusion_matrix(y_test_2018, prediction_linear).ravel()

# TPR/TNR rates
print('The TPR is:', str(tp) + '/' + str(tp + fn) + ',',
      round(recall_score(y_test_2018, prediction_linear) * 100, 2),'%')
print('The TNR is:', str(tn) + '/' + str(tn + fp) + ',',
    round(tn / (tn + fp) * 100, 2),'%')


# 4)
# create a gaussian SVM classifier
svm_classifier_rbf = svm.SVC(kernel = 'rbf')

# fit the classifier on training data
svm_classifier_rbf.fit(x_train_2017, y_train_2017)

# Predict using 2018 feature data
prediction_rbf = svm_classifier_rbf.predict(x_test_2018)

# calculate error rate
accuracy_rate_rbf = 100-(round(np.mean(prediction_rbf != y_test_2018) * 100, 2))

# Print accuracy rate
print('The gaussian SVM classifier has an accuracy of', accuracy_rate_rbf,'%')


# 5)
# create a polynomial SVM classifier
svm_classifier_poly = svm.SVC(kernel = 'poly')

# fit the classifier on training data
svm_classifier_poly.fit(x_train_2017, y_train_2017)

# Predict using 2018 feature data
prediction_poly = svm_classifier_poly.predict(x_test_2018)

# calculate error rate
accuracy_rate_poly = 100-(round(np.mean(prediction_poly != y_test_2018) * 100, 2))

# Print accuracy rate
print('The polynomial SVM classifier has an accuracy of', accuracy_rate_poly,'%')


# 6)
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
    for day in range(0, len(prediction_linear)):
        # Sell should occur on the last day of a green week at 
        # the adjusted_close price. Since i is tracking the current
        # trading week we need to minus 1 to get the adjusted close price
        # from the previous trading week
        if prediction_linear[day] == 0 and shares > 0:
            wallet = round(shares * adj_close[day - 1], 2)
            shares = 0
            
        # Buy should occur on the first day of a green week at the open price
        if prediction_linear[day] == 1 and shares == 0: 
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
# This method would close the year at $ 236.89 a profit of $ 136.89
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