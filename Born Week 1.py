# -*- coding: utf-8 -*-
"""
CS 677
Homework week 1
@author: Eric Born
"""
# run this !pip install pandas_datareader

# Using the stock ticker BSX - Boston Scientific Corporation.

import os

ticker = 'BSX'
input_dir = r'C:\Users\TomBrody\Desktop\School\677\wk1'
ticker_file = os.path.join(input_dir, ticker + '.csv')

try:
    with open(ticker_file) as f:
        lines = f.read().splitlines()
        # list_lines = lines.split('\n')
    print('opened file for ticker: ', ticker)
    """    your code for assignment 1 goes here
    """

except Exception as e:
    print(e)
    print('failed to read stock data for ticker: ', ticker)

# Initialize wallet and shares to track current money and number of shares.
wallet = 100.00
shares = 0

# Create empty lists for the open and close prices.
open_price = []
close_price = []
adj_close = []
long_ma = []
return_pct = []

# Use enumerate to find the indices where each comma starts.
# This allows us to count over to the start of the proper column for the
# different daily stock values.
# Loop starts from row 1 instead of 0, since 0 contains the headers.
for line in range(1, len(lines)):
    indices = [i for i, s in enumerate(lines[line]) if ',' in s]
    open_price.append(float(lines[line][indices[6] + 1:indices[7]]))
    close_price.append(float(lines[line][indices[9] + 1:indices[10]]))
    adj_close.append(float(lines[line][indices[11] + 1:indices[12]]))
    long_ma.append(float(lines[line][indices[-1] + 1:len(lines[line])]))

######
# Strategy 1
######

# Buy shares on the first day
# 8.319467554076539 shares purchased with $100.00
shares = wallet / open_price[31]
wallet = 0

for i in range(len(open_price) - 1):
    # SELL
    if close_price[i] > close_price[i + 1] and shares > 0:
        wallet = shares * adj_close[i]
        shares = 0
        # BUY
    if close_price[i] < close_price[i + 1] and shares == 0:
        shares = wallet / adj_close[i]
        wallet = 0

    # Question 1
# Currently own 4989.159389271963 shares Worth $176316.89
print('Question 1: ' + 'Currently own ' + str(shares) + ' shares' + '\n' +
      'Worth ' + '$' + str(round(shares * close_price[-1], 2)))

# Question 2
# Currently own 8.319467554076539 shares Worth $294.01
hold_stocks = 100 / open_price[0]
hold_worth = hold_stocks * close_price[-1]
print('Question 2: ' + 'Currently own ' + str(hold_stocks) + ' shares' + '\n' +
      'Worth ' + '$' + str(round(hold_worth, 2)))


######
# Strategy 2
######
wallet = 100.00
shares = 0

shares = wallet / open_price[0]
wallet = 0

for i in range(len(open_price) - 1):
    # SELL
    if adj_close[i] < long_ma[i] and shares > 0:
        wallet = shares * adj_close[i]
        shares = 0

        # BUY
    if adj_close[i] > long_ma[i] and shares == 0:
        shares = wallet / adj_close[i]
        wallet = 0

    # Strategy 2
# Currently own 0 shares Worth $0.0 and $163.21 in the wallet.
print('Strategy 2: ' + 'Currently own ' + str(shares) + ' shares' + '\n' +
      'Worth ' + '$' + str(round(shares * close_price[-1], 2))
      + ' and $' + str(round(wallet, 2)) + ' in the wallet.')


######
# Strategy 3
######
# Create functions to find the median, average, minimum and maximum value.
def median(lst):
    n = len(lst)
    s = sorted(lst)
    return (sum(s[n // 2 - 1:n // 2 + 1]) / 2.0, s[n // 2])[n % 2] if n else None


def avg(lst):
    return sum(lst) / len(lst)


def minimum(day):
    x = 0
    for i in range(len(day)):
        if day[i] < x:
            x = day[i]
    return x


def maximum(day):
    x = 0
    for i in range(len(day)):
        if day[i] > x:
            x = day[i]
    return x


# Initialize a list for each day of the week the stock market is open.
monday = []
tuesday = []
wednesday = []
thursday = []
friday = []

# For loop that grabs the stocks closing value on each day of the week and
# stores it into its own list based upon the day of the week.
for line in range(len(lines)):
    indices = [i for i, s in enumerate(lines[line]) if ',' in s]
    day = lines[line][indices[3] + 1:indices[4]]
    value = lines[line][indices[12] + 1:indices[13]]
    if day == 'Monday':
        monday.append(float(value))
    elif day == 'Tuesday':
        tuesday.append(float(value))
    elif day == 'Wednesday':
        wednesday.append(float(value))
    elif day == 'Thursday':
        thursday.append(float(value))
    elif day == 'Friday':
        friday.append(float(value))

# Output day data.
print('Day of the week' + '   min' + '        max' + '        average' +
      '     median' + '%\n' + '     Monday     ' +
      str(round(minimum(monday) * 100, 2)) + '%      ' +
      str(round(maximum(monday) * 100, 2)) + '%      ' +
      str(round(avg(monday) * 100, 4)) + '%    ' +
      str(round(median(monday) * 100, 4)) + '%\n' + '     Tuesday    ' +
      str(round(minimum(tuesday) * 100, 4)) + '%     ' +
      str(round(maximum(tuesday) * 100, 4)) + '%     ' +
      str(round(avg(tuesday) * 100, 4)) + '%     ' +
      str(round(median(tuesday) * 100, 4)) + '%\n' + '     Wednesday  ' +
      str(round(minimum(wednesday) * 100, 4)) + '%    ' +
      str(round(maximum(wednesday) * 100, 4)) + '%    ' +
      str(round(avg(wednesday) * 100, 4)) + '%     ' +
      str(round(median(wednesday) * 100, 4)) + '%\n' + '     Thursday   ' +
      str(round(minimum(thursday) * 100, 4)) + '%    ' +
      str(round(maximum(thursday) * 100, 4)) + '%     ' +
      str(round(avg(thursday) * 100, 4)) + '%     ' +
      str(round(median(thursday) * 100, 4)) + '%\n' + '     Friday     ' +
      str(round(minimum(friday) * 100, 4)) + '%    ' +
      str(round(maximum(friday) * 100, 4)) + '%     ' +
      str(round(avg(friday) * 100, 4)) + '%     ' +
      str(round(median(friday) * 100, 4)) + '%')

# I would pick Wednesday as it had the highest average return at 0.3% and also had the highest maximum gain of 12.39%.

######
# Strategy 4
######

# Initialize lists for each month that will be used to group the closing price
jan = []
feb = []
mar = []
apr = []
may = []
jun = []
july = []
aug = []
sept = []
octo = []
nov = []
dec = []

# For loop that grabs the stocks closing value on each day of the week and
# stores it into its own list based upon the month.
for line in range(len(lines)):
    indices = [i for i, s in enumerate(lines[line]) if ',' in s]
    month = lines[line][indices[1] + 1:indices[2]]
    value = lines[line][indices[12] + 1:indices[13]]
    if month == '1':
        jan.append(float(value))
    elif month == '2':
        feb.append(float(value))
    elif month == '3':
        mar.append(float(value))
    elif month == '4':
        apr.append(float(value))
    elif month == '5':
        may.append(float(value))
    elif month == '6':
        jun.append(float(value))
    elif month == '7':
        july.append(float(value))
    elif month == '8':
        aug.append(float(value))
    elif month == '9':
        sept.append(float(value))
    elif month == '10':
        octo.append(float(value))
    elif month == '11':
        nov.append(float(value))
    elif month == '12':
        dec.append(float(value))

# Output month data
# I would chose January as the best month to buy and hold.
# November would be the worst month to buy and hold.
print('Month' + '          min' + '        max' + '        average' +
      '     median' + '%\n' + 'January     ' +
      str(round(minimum(jan) * 100, 2)) + '%        ' +
      str(round(maximum(jan) * 100, 2)) + '%       ' +
      str(round(avg(jan) * 100, 4)) + '%     ' + str(round(median(jan) * 100, 4))
      + '%\n' + 'February    ' + str(round(minimum(feb) * 100, 4)) + '%     ' +
      str(round(maximum(feb) * 100, 4)) + '%     ' +
      str(round(avg(feb) * 100, 4)) + '%     ' + str(round(median(feb) * 100, 4)) +
      '%\n' + 'March     ' + str(round(minimum(mar) * 100, 4)) + '%    ' +
      str(round(maximum(mar) * 100, 4)) + '%    ' + str(round(avg(mar) * 100, 4)) +
      '%     ' + str(round(median(mar) * 100, 4)) + '%\n' + 'April      ' +
      str(round(minimum(apr) * 100, 4)) + '%    ' +
      str(round(maximum(apr) * 100, 4)) + '%     ' + str(round(avg(apr) * 100, 4)) +
      '%     ' + str(round(median(apr) * 100, 4)) + '%\n' + 'May        ' +
      str(round(minimum(may) * 100, 4)) + '%    ' +
      str(round(maximum(may) * 100, 4)) + '%     ' +
      str(round(avg(may) * 100, 4)) + '%     ' + str(round(median(may) * 100, 4)) +
      '%\n' + 'June        ' + str(round(minimum(jun) * 100, 2)) + '%      ' +
      str(round(maximum(jun) * 100, 2)) + '%      '
      + str(round(avg(jun) * 100, 4)) + '%    ' + str(round(median(jun) * 100, 4)) +
      '%\n' + 'July       ' + str(round(minimum(july) * 100, 4)) + '%     ' +
      str(round(maximum(july) * 100, 4)) + '%     ' +
      str(round(avg(july) * 100, 4)) + '%     ' + str(round(median(july) * 100, 4)) +
      '%\n' + 'August     ' + str(round(minimum(aug) * 100, 4)) + '%    ' +
      str(round(maximum(aug) * 100, 4)) + '%    ' + str(round(avg(aug) * 100, 4)) +
      '%     ' + str(round(median(aug) * 100, 4)) + '%\n' + 'September   ' +
      str(round(minimum(sept) * 100, 4)) + '%    ' + str(round(maximum(sept) * 100, 4)) +
      '%     ' + str(round(avg(sept) * 100, 4)) + '%     ' +
      str(round(median(sept) * 100, 4)) + '%\n' + 'October     ' +
      str(round(minimum(octo) * 100, 4)) + '%    ' + str(round(maximum(octo) * 100, 4)) +
      '%     ' + str(round(avg(octo) * 100, 4)) + '%     ' +
      str(round(median(octo) * 100, 4)) + '%\n' + 'November   ' +
      str(round(minimum(nov) * 100, 4)) + '%    ' + str(round(maximum(nov) * 100, 4)) +
      '%     ' + str(round(avg(nov) * 100, 4)) + '%     ' +
      str(round(median(nov) * 100, 4)) + '%\n' + 'December     ' +
      str(round(minimum(dec) * 100, 4)) + '%    ' + str(round(maximum(dec) * 100, 4)) +
      '%     ' + str(round(avg(dec) * 100, 4)) + '%     ' +
      str(round(median(dec) * 100, 4)) + '%')


