import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', 30)
pd.set_option('display.width', 150)

df = pd.read_csv('test/lesson_3_data.csv', encoding='windows-1251')
# print(df.head())

user_df = df[['tc', 'art_sp']]
user_df = user_df.rename(columns={'tc': 'user_id', 'art_sp': 'brand_info'})
user_df['brand_name'] = user_df.brand_info.apply(lambda s: s.split()[-1])

# users_purchases = user_df.groupby('user_id', as_index=False).brand_name.count()
users_purchases = user_df.groupby('user_id', as_index=False) \
    .agg(purchases=('brand_name', 'count')) \
    .query('purchases >= 5')

users_unique_brands = user_df.groupby('user_id', as_index=False) \
    .agg(unique_brands=('brand_name', pd.Series.nunique)) \
# print(users_unique_brands)

lovely_brand_purchases_df = user_df.groupby(['user_id', 'brand_name'], as_index=False) \
      .agg(purchases=('brand_info', 'count')) \
      .sort_values(['user_id', 'purchases'], ascending=[False, False]) \
      .groupby('user_id') \
      .head(1) \
      .rename(columns={'brand_name': 'lovely_brand',
                       'purchases': 'lovely_brand_purchases'})
# print(lovely_brand_purchases_df)

loyalty_df = users_purchases\
    .merge(users_unique_brands, on='user_id')\
    .merge(lovely_brand_purchases_df, on='user_id')

loyal_users = loyalty_df[loyalty_df.unique_brands == 1]
loyalty_df['loyalty_score'] = loyalty_df.lovely_brand_purchases / loyalty_df.purchases

#ax = sns.displot(loyalty_df.loyalty_score)

brand_loyality = loyalty_df.groupby('lovely_brand', as_index=False)\
                           .agg({'loyalty_score': 'median', 'user_id': 'count'})

ax = sns.barplot(x='lovely_brand', y='loyalty_score', data=brand_loyality)
plt.show()
