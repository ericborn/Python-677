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
import seaborn as sns
import scipy
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix


# setup input directory and filename
input_dir = r'C:\Users\TomBrody\Desktop\School\677\wk3\logistic regression'
filename = os.path.join(input_dir,'BSX-labeled.csv')

# read csv file into dataframe
df = pd.read_csv(filename)

# Create class column where red = 0 and green = 1
df['class'] = df['label'].apply(lambda x: 1 if x =='green' else 0)

# create separate dataframes for 2017 and 2018 data
df_2017 = df.loc[df['td_year']==2017]
df_2018 = df.loc[df['td_year']==2018]

# reset indexes
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

# Define features and class labels
features = ['mu', 'sig']
class_labels = ['green', 'red']

# create x training and test sets from 2017/2018 features data
x_train = df_2017_reduced[features].values
x_test = df_2018_reduced[features].values

# define label encoder
le = LabelEncoder()

# use label encoder and label values to develop y training and test sets
#y_train = le.fit_transform(df_2017_reduced['label'].values)
#y_test = le.fit_transform(df_2018_reduced['label'].values)

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



###############################

X = df_2017_reduced[features].values

y = df_2017_reduced['label'].values

X = np.c_[np.ones((X.shape[0], 1)), X]

green = df_2017_reduced.loc[y == 1]

red = df_2017_reduced.loc[y == 0]

plt.scatter(green.iloc[:, 1], green.iloc[:, 2], s=10, label='Green')
plt.scatter(red.iloc[:, 1], red.iloc[:, 2], s=10, label='Red')
plt.legend()

# prepare data for fitting
X = np.c_[np.ones((X.shape[0], 1)), X]
y = y[:, np.newaxis]
theta = np.zeros((X.shape[0], 1))

# Create functions to compute cost
def sigmoid(x):
    # Activation function used to map any real value between 0 and 1
    return 1 / (1 + np.exp(-x))

def net_input(theta, x):
    # Computes the weighted sum of inputs
    return np.dot(x, theta)

def probability(theta, x):
    # Returns the probability after passing through sigmoid
    return sigmoid(net_input(theta, x))

# Create cost and gradient functions
def cost_function(self, theta, x, y):
    # Computes the cost function for all the training samples
    m = x.shape[0]
    total_cost = -(1 / m) * np.sum(
        y * np.log(probability(theta, x)) + (1 - y) * np.log(
            1 - probability(theta, x)))
    return total_cost

def gradient(self, theta, x, y):
    # Computes the gradient of the cost function at the point theta
    m = x.shape[0]
    return (1 / m) * np.dot(x.T, sigmoid(net_input(theta,   x)) - y)

def fit(self, x, y):
    opt_weights = fmin_tnc(func=cost_function, #x0=theta,
                  fprime=gradient,args=(x, y.flatten()))
    return opt_weights[0]

parameters = fit(X, y)



x_values = [np.min(x_2018_test[:, 0] - 5), np.max(x_2018_test[:, 1] + 5)]
y_values = - (x_2018_test[0] + np.dot(x_2018_test[1], x_values))










#####################

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
