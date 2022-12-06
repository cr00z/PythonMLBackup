# Exercise 2.2.5

import datetime

year, month, date = input().split(' ')
time_delta = int(input())
dt = datetime.date(int(year), int(month), int(date))

newdt = dt + datetime.timedelta(time_delta)
print(newdt.year, newdt.month, newdt.day)