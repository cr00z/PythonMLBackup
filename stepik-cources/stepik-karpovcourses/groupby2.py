import pandas as pd

pd.set_option('display.max_columns', 30)
pd.set_option('display.width', 150)

sales = pd.read_excel('test/2018_Sales_Total_v2.xlsx')
print(sales.set_index('date').resample('M')['ext price'].sum())
print(sales.set_index('date').groupby('name').resample('M')['ext price'].sum())
month_grouper = pd.Grouper(key='date', freq='M')
year_grouper = pd.Grouper(key='date', freq='A-DEC')
print(sales.groupby(['name', month_grouper])['ext price'].sum())
print(sales.groupby(['name', year_grouper])['ext price'].sum())

daily_sales = sales.groupby([pd.Grouper(key='date', freq='D')]) \
    .agg(daily_sales=('ext price', 'sum')).reset_index()
daily_sales['quarter_sales'] = daily_sales\
    .groupby(pd.Grouper(key='date', freq='Q')).agg({'daily_sales': 'cumsum'})
print(daily_sales[daily_sales['date'].between('2018-03-29', '2018-04-04')])
