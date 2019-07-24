# -*- coding: utf-8 -*-
"""
CS 677
Homework week 2
@author: eborn
"""

# transactions from a bakery
import os
import pandas as pd
import math
import numpy as np 
import random

input_dir = r'C:\Users\eborn\Desktop\School\677\wk2'
input_file  = os.path.join(input_dir, 'BreadBasket_DMS.csv')
output_file  = os.path.join(input_dir, 'BreadBasket_DMS_output.csv')

def compute_period(hour):
    if hour in range(0, 6):
        return 'night'
    elif hour in range(6, 12):
        return 'morning'
    elif hour in range(12, 18):
        return 'afternoon'
    elif hour in range(18, 24):
        return 'evening'
    else:
        return 'unknown'    

df = pd.read_csv(input_file)

items = set(df['Item'])
price_dict = dict()
price_list = list(np.linspace(0.99, 10.99, 100))
for x in items:
    price_dict[x] = random.choice(price_list)
    
df['Item_Price'] = df['Item'].map(price_dict)
df['Item_Price'] = df['Item_Price'].round(2)


df['Date'] = pd.to_datetime(df['Date'])
df['Time'] = pd.to_datetime(df['Time'])
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year 
df['Day'] = df['Date'].dt.day
df['Weekday'] = df['Date'].dt.weekday_name 

df['Hour'] = df['Time'].dt.hour
df['Min'] = df['Time'].dt.minute
df['Sec'] = df['Time'].dt.second

df['Period'] = df['Hour'].apply(compute_period)

df.reset_index(level=0)
col_list = ['Year','Month','Day','Weekday', 'Period', 
            'Hour','Min','Sec',
            'Transaction','Item','Item_Price']
df = df[col_list]

df.to_csv(output_file, index=False)

# 1)
# using groupby on specified period of time and transaction along with nlargest 
# to find the busiest hour, weekday and period of the day.

# a)
# 11 is the busiest hour with 1445 transactions.
print(df.groupby('Hour')['Transaction'].nunique().nlargest(1))
print('11 is the busiest hour with 1445 transactions.')

# b)
# busiest day of the week is Saturday with 2068 transactions.
print(df.groupby('Weekday')['Transaction'].nunique().nlargest(1))
print('Saturday is the busiest day of the week with 2068 transactions.')

# c)
# Busiest period of the day is afternoon with 5307 transactions.
print(df.groupby('Period')['Transaction'].nunique().nlargest(1))
print('Afternoon is the busiest period of the day with 5307 transactions.')

# 2)
# using groupby on specified period of time and Item_Price along with nlargest 
# to find the most profitable hour, weekday and period of the day.

# a)
# 11 is the most profitable hour with $21453.44 worth of transactions.
print(df.groupby('Hour')['Item_Price'].sum().nlargest(1))
print('11 is the most profitable hour with $21453.44 worth of transactions.')

# b)
# Saturday is the most profitable day of the week with $31531.83 worth of transactions.
print(df.groupby('Weekday')['Item_Price'].sum().nlargest(1))
print('Saturday is the most profitable day of the week with $31531.83 worth of transactions.')

# c)
# Afternoon is the most profitable period of the day with $81299.97 worth of transactions.
print(df.groupby('Period')['Item_Price'].sum().nlargest(1))
print('Afternoon is the most profitable period of the day with $81299.97 worth of transactions.')

# 3)
# using value_counts on Item along with nlargest and nsmallest to find the
# most and least popular items.

print(df['Item'].value_counts().nlargest(3))
print("Coffee, Bread and Tea are the most popular items.")

print(df['Item'].value_counts().nsmallest(8))
print("The least popular items are The BART, Raw bars and Polenta, Chicken sand, "
       "Adjustment, Gift voucher, Bacon and Olum & polenta")

# 4)
# Using groupby on the different time columns along with transactions to find
# the total number of transactions per day of the week for the entire dataset
trans_day = pd.DataFrame(df.groupby(['Year', 'Month', 'Day', 'Weekday'])['Transaction'].nunique())

print("Number of barristas required per day:")
# Find the max number of transactions for each day of the week.
max_day = trans_day.groupby('Weekday')['Transaction'].max().to_frame()

# using .ceil to round up, find the number of barristas required by dividing
# The total number of transactions per day by 50, which is the max they can
# handle. Set as type int to maintain whole numbers since people cannot be
# represented by a float.
barristas = (np.ceil(max_day.Transaction/50)).astype(int)
print("Maximum Barristas per day:")
print(barristas)

# 5)
# Creates a new column for the category of the item. Starts all values as unknown
df['category'] = 'unknown'

# Creates lists for food and drink items
food = ['Bread', 'Muffin', 'Pastry', 'Cookies', 'Fudge', 'Soup', 'Cake', 
        'Chicken sand', 'Sandwich', 'Eggs', 'Brownie', 'Granola', 'Empanadas',
        'Bread Pudding', 'Truffles', 'Bacon', 'Kids biscuit', 'Caramel bites',
        'Toast', 'Scone','Crepes','Vegan mincepie','Bare Popcorn','Muesli',
        'Crisps','Brioche and salami', 'Salad', 'Chicken Stew', 
        'Spanish Brunch', 'Raspberry shortbread sandwich', 
        'Extra Salami or Feta','Duck egg','Baguette']

drink = ['Coke', 'Hot chocolate', 'Coffee', 'Tea', 'Mineral water', 'Juice',
         'Smoothies']

# Checkes for item column containing value from the food or drink lists
# If an item is found, the category column value is replaced with the 
# string 'food' or 'drink'
df.loc[df['Item'].isin(food), 'category'] = 'food'
df.loc[df['Item'].isin(drink), 'category'] = 'drink'

# Create average function that rounds to 2 decimal places.
def avg(items):
    return round(sum(items) / len(items), 2)

# Calculate food and drink averages
avg_food = avg(df.loc[df['category'] == 'food', 'Item_Price'].values)
avg_drink = avg(df.loc[df['category'] == 'drink', 'Item_Price'].values)

# Average price of food and drink item
print('The average price of food is: $' + str(avg_food))
print('The average price of a drink is: $' + str(avg_drink))

# 6)
# Total amounts made in food and drink sales
# Food sales are higher at $76,660 vs $58,365
food_sales = round(df.loc[df['category'] == 'food', 'Item_Price'].sum(), 2)
drink_sales = round(df.loc[df['category'] == 'drink', 'Item_Price'].sum(), 2)
print('Total food sales: $' + str(food_sales))
print('Total drink sales: $' + str(drink_sales))

# 7)
# Creates counts based up groups by day of the week and item
top_five = df.groupby('Weekday')['Item'].value_counts()

# Output top 5 per day
top_five.groupby('Weekday').nlargest(5)

# #1 on all days is coffee
# #2 on all days is bread
# #3 on all days is tea
# #4 on Monday is pastry. Tues, Weds, Thurs, Sat and Sun its cake. Fri is sandwich
# #5 on Monday is sandwich. Tues and Thurs its pastry. Fri is cake. All other days are none.

# 8)
# Output bottom 5 per day
top_five.groupby('Weekday').nsmallest(5)

# Since there are many items with only 1 sale per day, I don't think its really
# fair to compare the items in this fashion.

# 9)
# Find the count of each category when grouping by transaction number
drink_counts = df.loc[df['category'] == 'drink', 'Transaction'].value_counts()

# Output transactions with drinks
drink_counts.nlargest(8)
drink_counts.nsmallest(3954)
avg(drink_counts)
