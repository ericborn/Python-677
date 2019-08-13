# -*- coding: utf-8 -*-
"""
Eric Born
Class: CS677 - Summer 2
Date: 12 Aug 2019
Final project
Predicting the winning team in the video game League of Legends
"""

import os
from sys import exit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.metrics import confusion_matrix, recall_score
#from sklearn.model_selection import train_test_split
#from sklearn.model_selection import StratifiedKFold

sns.set_style("darkgrid")

# setup input directory and filename
data = 'games'
input_dir = r'C:\Users\TomBrody\Desktop\School\677\Final'
ticker_file = os.path.join(input_dir, data + '.csv')

# read csv file into dataframe
try:
    lol_df = pd.read_csv(ticker_file)
    print('opened file for ticker: ', data,'\n')

except Exception as e:
    print(e)
    exit('failed to read LoL data from: '+ str(data)+'.csv')

# describe the total rows and columns
print('The total length of the dataframe is', lol_df.shape[0], 'rows',
      'and the width is', lol_df.shape[1], 'columns')

# create a class label using 0 or 1 to indicate winning team
# 0 = team 1 won
# 1 = team 2 won
lol_df['win'] = lol_df['winner'].apply(lambda x: 0 if x == 1 else 1)

# remove columns gameId, creationTime, seasonId and winner
lol_df.drop(lol_df.columns[[0,1,3,4]], 
                     axis = 1, inplace = True)

# there are -1's stored in the t1 and t2 ban columns that need to
# be replaced before the chi-squared can be run
# for loop cycles through t1_ban1, t1_ban2, etc. for both teams
# and sets the row to a 0 instead of a -1
for team in range(1, 3):
    for char in range(1, 5):
        t = 't'+str(team)+'_ban'+str(char)
        print(t, 'had', len(lol_df.loc[lol_df[t] == -1, t]),
              '-1s replaced with a 0')
        lol_df.loc[lol_df[t] == -1, t] = 0

# view row 89, all columns
#lol_df.iloc[89,:]

# store wins in variable for each time
team1 = sum(lol_df.win == 0)
team2 = sum(lol_df.win == 1)

# create a ratio of the wins
ratio = round(team1 / team2, 4)
print('\nTeam win ratios\n','team 1 : team 2\n', str(ratio)+' :   1')

# 58 columns remaining
lol_df.head()

# store dataframe values into an array for faster computation
lol_array = lol_df.values

lol_array[:,-1]

# x stores all columns except for the win column
chi_x = lol_array[:,0:57]

# y stores only the win column since its used as a predictor
chi_y = lol_array[:,57]

# implement chi-squared to evaluate feature significance
chi_test = SelectKBest(score_func = chi2, k=4)
chi_fit = chi_test.fit(chi_x, chi_y)



# split dataset into 33% test 66% training
#lol_df_train, lol_df_test = train_test_split