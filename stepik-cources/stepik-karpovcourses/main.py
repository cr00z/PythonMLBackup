import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', 30)
pd.set_option('display.width', 150)

user_data = pd.read_csv('test/user_data.csv')
logs = pd.read_csv('test/logs.csv')
# print(user_data.shape)
# print(user_data.info())
# print(user_data.describe())
# print(user_data.isna().sum())
# print(logs.shape)
# print(logs.info())
# print(logs.isna().sum())

# print(logs.platform.nunique())

print(user_data.columns)
print(logs.columns)
# best_client = logs\
#     .groupby('client', as_index=False)\
#     .agg({'success': 'sum'})\
#     .groupby('success')\
#     .agg({'client': set}) \
#     .sort_values('success', ascending=False)\
#     .head(1)\
#     .client.to_list()[0]
# print(*sorted(best_client), sep=', ')
#
#bcs = logs.groupby('client').success.sum()
# best_client = list(bcs.loc[bcs == bcs.max()].index)
# print(*sorted(best_client), sep=', ')

# best_platform = logs.groupby('platform').success.sum()
# print(best_platform)
# best_platform = logs.loc[logs.success].value_counts('platform')
# print(best_platform)

# client_platform = user_data.loc[user_data.premium][['client', 'premium']]\
#     .merge(logs[['client', 'platform']], on='client')\
#     .value_counts('platform')
# print(client_platform)

# data = user_data.merge(logs)
# axf = sns.distplot(data[data.premium == False].age)
# axt = sns.distplot(data[data.premium == True].age)

# bcs = logs.loc[logs.success].groupby('client', as_index=False)\
#     .agg({'success': 'count'}).sort_values('success')\
#     .groupby('success', as_index=False)\
#     .client.count()
# print(bcs)
# ax = sns.barplot(x='success', y='client', data=bcs)

computer_success = logs\
    .query('success & platform == "computer"')\
    .merge(user_data, on='client')\
#    .groupby('age')\
#    .agg(purchases=('age', 'count'))\
#    .reset_index()
ax = sns.countplot(x='age', data=computer_success)
plt.show()