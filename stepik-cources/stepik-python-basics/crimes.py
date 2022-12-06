# Exercise 3.3.7

import pandas as pd
df = pd.read_csv('test/Crimes.csv')
df2015 = df[df['Date'].str.contains('2015')]
print(df2015['Primary Type'].value_counts().keys()[0])
