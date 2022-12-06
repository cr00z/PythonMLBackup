import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', 30)
pd.set_option('display.width', 150)


def read_n_agg(fn):
    return pd.read_csv(fn, sep=';').groupby('company').mean()

print(read_n_agg('https://stepik.org/media/attachments/lesson/359209/companies.csv'))


df = pd.DataFrame({'wealth': ['medium'], 'age': [64]})
print(df)
medium_35 = df[(df.wealth == 'medium') & (df.age > 35)]
unique_num = df.nunique().agg()
print(unique_num)

taxi = pd.read_csv(
    'test/taxi_peru.csv',
    sep=';', parse_dates=['start_at', 'end_at', 'arrived_at']
)
print(taxi.groupby('source').agg({'source': 'count'}) / len(taxi))
print(taxi.agg({'source': 'unique', 'source': 'count'}) / len(taxi))
print(taxi.head())
scores_num = taxi.driver_score.count()
driver_score_counts = taxi\
    .groupby('driver_score', )\
    .agg(percentage=('driver_score', lambda x: x.count() / scores_num))\
    .sort_values('driver_score')\
    .mul(100).round(2).reset_index()
print(driver_score_counts)

ax = sns.barplot(x='driver_score', y='percentage', data=driver_score_counts, color='blue', alpha=0.5)
ax.set(xlabel='Driver score', ylabel='Percentage')
sns.despine()  # убрать часть рамки графика

scores_num = taxi.rider_score.count()
rider_score_counts = taxi\
    .groupby('rider_score', )\
    .agg(percentage=('rider_score', lambda x: x.count() / scores_num))\
    .sort_values('rider_score')\
    .mul(100).round(2).reset_index()
print(rider_score_counts)

ax = sns.barplot(x='rider_score', y='percentage', data=rider_score_counts, color='orange', alpha=0.5)
ax.set(xlabel='Rider score', ylabel='Percentage')
sns.despine()  # убрать часть рамки графика
plt.show()
