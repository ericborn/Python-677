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
#import numpy as np
import pandas as pd
#import seaborn as sns
from sklearn.cluster import KMeans
#from sklearn.metrics import confusion_matrix, recall_score
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

# Create a separate dataframe for 2017 and 2018 data
bsx_df_years = bsx_df.loc[(bsx_df['td_year']==2017) | (bsx_df['td_year']==2018)]

# Reset indexes
bsx_df_years = bsx_df_years.reset_index(level=0, drop=True)

# Define features labels.
features = ['mu', 'sig']

# Create reduced dataframe only containing week number, mu, sig and label
bsx_reduced = pd.DataFrame( {'week nbr' : range(0, 53),
            'mu'    : bsx_df_years.groupby('td_week_number')['return'].mean(),
            'sig'   : bsx_df_years.groupby('td_week_number')['return'].std(),
            'label' : bsx_df_years.groupby('td_week_number')['class'].first()})

# Remove index name labels from dataframes
del bsx_reduced.index.name
    
# Create x and y training and test sets from 2017 data
x_values = bsx_reduced[features].values
y_values = bsx_reduced['label'].values

# Setup scaled x values
scaler = StandardScaler()
scaler.fit(x_values)
x_values_scaled = scaler.transform(x_values)

# create the k-means classifier with 3 clusters
kmeans_classifier = KMeans(n_clusters = 3)
y_means = kmeans_classifier.fit_predict(x_values_scaled)
centroids = kmeans_classifier.cluster_centers_

# create empty list to store inertia values
inertia_list = []

# for loop that iterates k 1 through 8
for k in range(1, 9):
    kmeans_classifier = KMeans(n_clusters = k)
    y_means = kmeans_classifier.fit_predict(x_values_scaled)
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
y_means = kmeans_classifier.fit_predict(x_values_scaled)
centroids = kmeans_classifier.cluster_centers_

# Generate a plot with points colored green/red depending on their cluster
fig, ax = plt.subplots(1,figsize =(7,5))
plt.scatter(x_values_scaled[y_means == 0, 0], x_values_scaled[y_means == 0, 1],
                s = 75, c ='red', label = 'red')
plt.scatter(x_values_scaled[y_means == 1, 0], x_values_scaled[y_means == 1, 1],
                s = 75, c = 'green', label = 'green')
plt.scatter(centroids[:, 0], centroids[:,1] ,
                s = 200 , c = 'black', label = 'Centroids')

# calculate total green and red points
total_green = sum(y_means == 1)
total_red =   sum(y_means == 0)

# print total and percent of each color
print('Total red points:', total_red, '\npercentage of red:', 
      round(total_red/len(y_means), 2)*100,
      '\nTotal green points:', total_green, '\npercentage of green', 
      round(total_green/len(y_means), 2)*100)
