# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 14:46:58 2018

@author: epinsky
"""

# sigmoid function
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix

# setup input directory and filename
input_dir = r'C:\Users\TomBrody\Desktop\School\677\wk3\logistic regression'
filename = os.path.join(input_dir,'BSX-labeled.csv')

# read csv file into dataframe
df = pd.read_csv(filename)

# Create class column where red = 0 and green = 1
df['class'] = df['label'].apply(lambda x: 1 if x=='green' else 0)

# create separate dataframes for 2017 and 2018 data
df_2017 = df.loc[df['td_year']==2017]
df_2018 = df.loc[df['td_year']==2018]

# reset indexes
df_2017 = df_2017.reset_index(level=0, drop=True)
df_2018 = df_2018.reset_index(level=0, drop=True)  

# Create reduced dataframe only containing week number, mu, sig and label
df_2017_reduced = pd.DataFrame( {'week nbr' : range(1, 53),
                        'mu'   : df_2017.groupby('td_week_number')['return'].mean(),
                        'sig'  : df_2017.groupby('td_week_number')['return'].std(),
                        'label': df_2017.groupby('td_week_number')['label'].first()})

# Create reduced dataframe only containing week number, mu, sig and label
df_2018_reduced = pd.DataFrame( {'week nbr' : range(0, 53),
                        'mu'   : df_2018.groupby('td_week_number')['return'].mean(),
                        'sig'  : df_2018.groupby('td_week_number')['return'].std(),
                        'label': df_2018.groupby('td_week_number')['label'].first()})

# Replacing nan in week 52 sigma column with a zero due to 
# there being only 1 trading day that week.
df_2018_reduced = df_2018_reduced.fillna(0)

# remove index name labels from dataframes
del df_2017_reduced.index.name
del df_2018_reduced.index.name

x_2017 = df_2017_reduced[['mu', 'sig']].values

scaler = StandardScaler()
scaler.fit(x_2017)
x_2017_train = scaler.transform(x_2017)
y_2017_train = df_2017_reduced['label'].values

log_reg_classifier = LogisticRegression()
log_reg_classifier.fit(x_2017_train, y_2017_train)
#new_x = scaler.transform(np.asmatrix ([6 , 160]))

#####
# build prediction data
x_2018 = df_2018_reduced[['mu', 'sig']].values

scaler = StandardScaler()
scaler.fit(x_2018)
x_2018_test = scaler.transform(x_2018)
y_2018_test = df_2018_reduced['label'].values

predicted = log_reg_classifier.predict(x_2018_test)
accuracy = log_reg_classifier.score(x_2018_test, y_2018_test)

labels=['Green', 'Red', 'Green', 'Red']

confusion_matrix(y_2018_test, predicted)

#########################



N = len(y)
learning_rate = 0.01

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def loss(h, y):
    return (-y*np.log(h)-(1-y)*np.log(1-h)).mean()

def add_intercept(X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

threshold = 0.1
n=5
iterations=10000
# initialize the weights

x = add_intercept(x)     
weights = np.zeros(x.shape[1])

def compute_weights(x,weights,iterations, learning_rate, debug_step=1000):
    for i in range(iterations):
        y_pred = np.dot(x, weights)
        phi = sigmoid(y_pred)
        gradient = np.dot(x.T, (phi-y))/N
        weights = weights - learning_rate * gradient
        if i % debug_step==0:
            y_pred = np.dot(x, weights)
            phi = sigmoid(y_pred)
            print('i:', i, 'loss: ', loss(phi, y_pred))
    print('rate: ', learning_rate, ' iterations ', iterations, ' weights: ', weights)
    return weights


# predict:

fig = plt.figure(figsize=(5, 5))
ax = plt.gca()


df = df_2017_reduced[df_2017_reduced['label']=='red']
plt.scatter(df_2017_reduced['mu'].values, df_2017_reduced['sig'].values, color='red',
            s= 100, label='Class 0')

df = df_2017_reduced[df_2017_reduced['label']=='green']
plt.scatter(df['mu'].values, df['sig'].values, color='green',
            s= 100, label='Class 1')

for i in range(len(df_2017_reduced)):
    x_text = df_2017_reduced['mu'].iloc[i] + 0.05
    y_text = df_2017_reduced['sig'].iloc[i] + 0.2
    id_text = df_2017_reduced['week nbr'].iloc[i]
    plt.text(x_text, y_text, str(id_text), fontsize=14)


plt.xlim(4,7)
plt.ylim(4,15)


h_1, h_2 = 4.5, 6.5


iterations=100
weights = np.zeros(x.shape[1])
weights = compute_weights(x,weights,iterations=iterations, learning_rate=learning_rate,
                          debug_step=25)
f_1 = (-weights[0]-weights[1]*h_1)/weights[2]
f_2 = (-weights[0]-weights[1]*h_2)/weights[2]
plt.plot([h_1,h_2], [f_1, f_2], color='gray', label=str(iterations) + ' iterations', 
         lw=1)

iterations=250
weights = np.zeros(x.shape[1])
weights = compute_weights(x,weights,iterations=iterations, learning_rate=learning_rate,
                          debug_step=100)
f_1 = (-weights[0]-weights[1]*h_1)/weights[2]
f_2 = (-weights[0]-weights[1]*h_2)/weights[2]
plt.plot([h_1,h_2], [f_1, f_2], color='magenta', label=str(iterations) + ' iterations', 
         lw=1)

iterations=1000
weights = np.zeros(x.shape[1])
weights = compute_weights(x,weights,iterations=iterations, learning_rate=learning_rate,
                          debug_step=1000)
f_1 = (-weights[0]-weights[1]*h_1)/weights[2]
f_2 = (-weights[0]-weights[1]*h_2)/weights[2]
plt.plot([h_1,h_2], [f_1, f_2], color='blue', label=str(iterations) + ' iterations', lw=2)



plt.xlabel('mu')
plt.ylabel('sig')
plt.legend(loc='upper left')
plt.text(5.5,14,'learn.rate = ' + str(learning_rate), fontsize=14)

root_name = 'logistic_regression_gradient_descent_'+str(learning_rate)
root_name = root_name.replace('.', '_')

filename = os.path.join(input_dir,root_name + '_new.pdf')
plt.savefig(filename)
plt.show()    











