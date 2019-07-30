# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 02:37:06 2019

@author: AvantikaDG and Eugene Pinsky
"""
import os
import pandas as pd
import numpy as np

# Import CSV
input_dir = r'C:\Users\TomBrody\Desktop\School\677\wk3'
input_file  = os.path.join(input_dir, 'tips.csv')
output_file  = os.path.join(input_dir, 'tips_output.csv')
df = pd.read_csv(input_file)

# Add column for tip percent of total bill
df['tip_percent'] = 100.0 * df['tip']/df['total_bill']

# Ouptut df to a new csv with added column
df.to_csv(output_file, index=False)

# 1)
# Average tip percent for lunch and dinner
print(round(df.groupby('time')['tip_percent'].mean(), 2))

# 2)
# Average tip percent for each day of the week
print(round(df.groupby('day')['tip_percent'].mean(), 2))

# 3)
# Highest tips by percentage across all days/times
print(round(df.groupby(['day', 'time'])['tip_percent'].max(), 2))

# Highest tips by overall dollar amount across all days/times
print(df.groupby(['day', 'time'])['tip'].max())
    
# 4)  
# correlation between meal prices and tips
correlation_tips_vs_meal = df.corr(method='pearson')['tip_percent']['total_bill']
correlation_tips_vs_meal = round(correlation_tips_vs_meal , 4)
print('The correlation between meal prices and tips is:', correlation_tips_vs_meal)
if correlation_tips_vs_meal > 0 :
    print('Tips increase with a higher bill amount. ')
elif correlation_tips_vs_meal < 0:
    print("Tips decrease with a higher bill amount. ")
else:
    print('There is no relationship between tips and bill amount ')

# 5)
# correlation between size of group and tips
correlation_tips_vs_group = df.corr(method='pearson')['tip_percent']['size']
correlation_tips_vs_group = round(correlation_tips_vs_group, 4)
print('\nCorrelation between tips and group size:', correlation_tips_vs_group)
if correlation_tips_vs_group > 0 :
    print('Tips increase for larger groups.')
elif correlation_tips_vs_group < 0:
    print('Tips decrease for larger groups.')
else:
    print('No relationship between tips and group size.')
   
# 6) 
# percent of people smoking
print("\nPercentage of smokers is", round(100*len(df[df.smoker == "Yes"])/len(df),2), "%")

# 7)
# Correlation between tips and time
time = list(range(len(df)))
correlation_tips_vs_time = df['tip_percent'].corr(pd.Series(time), method='pearson')
correlation_tips_vs_time = round(correlation_tips_vs_time,4)
print('\nCorrelation between tips and time:', correlation_tips_vs_time)
if correlation_tips_vs_time > 0 :
    print('Tips increase with time.')
elif correlation_tips_vs_time < 0:
    print('Tips decrease with time.')
else:
    print('No relationship between tips and time.')

# 8)
# correlations in tip amounts between smokers and non-smokers
mean_tip_smokers_df = df.groupby(['smoker']).mean()  
mean_tip_smoke_yes = mean_tip_smokers_df['tip_percent'][0]
mean_tip_smoke_no = mean_tip_smokers_df['tip_percent'][1]  

print('\naverage tip for non-smokers:', round(mean_tip_smoke_no, 2)) 
print('average tip for smokers:', round(mean_tip_smoke_yes, 2)) 

if mean_tip_smoke_no > mean_tip_smoke_yes:
    print("Non-smokers pay larger tips.")
elif mean_tip_smoke_no < mean_tip_smoke_yes:
    print("Smokers pay larger tips.")
else:
    print("Smokers and non-smokers pay equal tips")