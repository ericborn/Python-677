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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, recall_score

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
    exit('failed to read stock data for ticker: '+ str(data))

# describe the total rows and columns
print('The total length of the dataframe is', lol_df.shape[0], 'rows',
      'and the width is', lol_df.shape[1], 'columns')

# remove columns creationTime and seasonId
lol_df.drop(lol_df.columns[[1, 3]], 
                     axis = 1, inplace = True)

# print the number of wins by team 1 and team 2 
for win in range(1, 3):
    print('Total team', win, 'wins:', sum(lol_df.winner == win))




