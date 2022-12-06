from sparklines import sparklines
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import skew, mode
import functools

pd.set_option('display.max_columns', 30)
pd.set_option('display.width', 150)

df = sns.load_dataset('titanic')
print(df.columns)

print(df.agg({'fare': ['sum', 'mean'], 'sex': 'count'}))
print(df.agg(fare_sum=('fare', 'sum'),
             fare_mean=('fare', 'mean'),
             sex_count=('sex', 'count')))

agg_func_math = {
    'fare': ['sum', 'mean', 'median', 'min', 'max', 'std', 'var', 'mad', 'prod']
}
print(df.groupby('embark_town').agg(agg_func_math).round(2))
print(df.groupby('embark_town').agg({'fare': 'describe'}).round(2))
print(df.groupby('deck').agg({'embark_town': ['count', 'nunique', 'size']}))
print(df.sort_values('fare', ascending=False)
      .groupby('embark_town')
      .agg({'fare': ['first', 'last']}))
print(df.groupby('embark_town').agg({'fare': ['max', 'min']}))
print(df.groupby('embark_town').agg({'fare': ['idxmax', 'idxmin']}))
print(df.loc[[258, 378]])
print(df.groupby('embark_town').agg({'fare': [skew, mode, pd.Series.mode]}))
print(df.groupby('class').agg({'deck': ['nunique', mode, set]}))

q_25 = functools.partial(pd.Series.quantile, q=0.25)
q_25.__name__ = '25%'
lambda_25 = lambda x: x.quantile(.25)
lambda_25.__name__ = 'q25'
#print(df.groupby('embark_town').agg({'fare': [q_25, lambda_25]}))


def count_null(x):
    return x.size - x.count()


def uniqie_nan(s):
    return s.nunique(dropna=False)


agg_func_custom = {
    'embark_town': ['count', 'nunique', 'size', uniqie_nan, count_null, set]
}
print(df.groupby('deck').agg(agg_func_custom))


def sparkline_str(x):
    bins = np.histogram(x)[0]
    return ''.join(sparklines(bins))


agg_func_largest = {'fare': ['max', sparkline_str]}
print(df.groupby(['class', 'embark_town']).agg(agg_func_largest))

print(df.groupby(['embark_town', 'class'])
        .agg({'fare': 'sum'})
        .assign(pct_total=lambda x: x / x.sum()))
