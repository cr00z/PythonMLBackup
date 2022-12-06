import pandas as pd

df = pd.read_csv('test/lesson_1_data.csv', encoding='windows-1251', sep=";")
df = df.rename(columns={'Номер': 'number',
                        'Дата создания': 'create_date',
                        'Дата оплаты': 'payment_date',
                        'Title': 'title',
                        'Статус': 'status',
                        'Заработано': 'amount',
                        'Город': 'city',
                        'Платежная система': 'payment_system'})
all_money = df.amount.sum()
print(all_money)
money_by_city = df \
    .query("status == 'Завершен'") \
    .groupby(['title'], as_index=False) \
    .aggregate({'amount': 'sum', 'number': 'count'}) \
    .sort_values('amount', ascending=False) \
    .rename(columns={'number': 'success_order'})
print(money_by_city)
money_by_city.to_csv('test/money_to_city.csv', index=False)
