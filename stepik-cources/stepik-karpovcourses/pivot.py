import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 30)
pd.set_option('display.width', 150)

df = pd.read_excel('test/salesfunnel.xlsx')

df.Status = df.Status.astype('category')
df.Status.cat.set_categories(
    ['won', 'pending', 'presented', 'declined'],
    inplace=True)

print(pd.pivot_table(df, index='Name'))
print(pd.pivot_table(df, index=['Manager', 'Rep'],
                     values='Price',
                     aggfunc=['mean', len]))
print(pd.pivot_table(df, index=['Manager', 'Rep'],
                     columns='Product',
                     values=['Price', 'Quantity'],
                     aggfunc=sum,
                     fill_value=0))
print(pd.pivot_table(df, index=['Manager', 'Rep', 'Product'],
                     values=['Price', 'Quantity'],
                     aggfunc=sum,
                     fill_value=0,
                     margins=True))
table = pd.pivot_table(df, index=['Manager', 'Status'],
                       columns=["Product"],
                       values=['Quantity', 'Price'],
                       aggfunc={'Quantity': len, 'Price': np.sum},
                       fill_value=0,
                       margins=True)
print(table.query('Manager == "Debra Henley"'))
