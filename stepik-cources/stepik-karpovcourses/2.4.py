import pandas as pd

# pd.set_option('display.width', 320)
pd.set_option('display.max_columns', 30)

bookings = pd.read_csv('test/bookings.csv', sep=';')
bookings_head = bookings.head(7)

# print(bookings.shape[1])
# print(bookings.dtypes.value_counts())
# print(bookings.info())

# Приведите названия колонок к нижнему регистру и замените пробелы на знак
# нижнего подчеркивания
bookings.rename(columns=lambda x: x.lower().replace(' ', '_'), inplace=True)
print(bookings.columns)

# Пользователи из каких стран совершили наибольшее число успешных бронирований?
# Укажите топ-5.
users = bookings \
    .query("is_canceled == 0") \
    .groupby('country', as_index=False) \
    .agg({'hotel': 'count'}) \
    .sort_values(['hotel'], ascending=False) \
    .rename(columns={'hotel': 'arrivals'})
users = bookings \
    .query("is_canceled == 0") \
    .groupby('country') \
    .size() \
    .sort_values(ascending=False)
users = bookings[bookings['is_canceled'] == 0].country.value_counts()
users = bookings.query('is_canceled == 0').value_counts('country')
users = bookings.loc[bookings['is_canceled'] == 0]['country'].value_counts()
# print(users.head())

# На сколько ночей в среднем бронируют отели разных типов?
nights = bookings \
    .groupby('hotel', as_index=False) \
    .agg({'stays_total_nights': 'mean'})
# print(nights)

# Иногда тип номера, полученного клиентом (assigned_room_type), отличается
# от изначально забронированного (reserved_room_type). Такое может произойти,
# например, по причине овербукинга. Сколько подобных наблюдений встретилось
# в датасете?
overbooking = bookings \
    .query('assigned_room_type != reserved_room_type') \
    .count()[0]
overbooking = bookings \
    .query('assigned_room_type != reserved_room_type') \
    .shape[0]
# print(overbooking)

# На какой месяц чаще всего успешно оформляли бронь в 2016? Изменился ли самый
# популярный месяц в 2017?
month_2016 = bookings \
    .query("arrival_date_year == 2016") \
    .arrival_date_month \
    .value_counts()
# print(month_2016.head(1))
month_2017 = bookings \
    .query("arrival_date_year == 2017") \
    .groupby(['arrival_date_month']) \
    .agg({'hotel': 'count'}) \
    .sort_values('hotel', ascending=False)
# print(month_2017.head(1))

months = bookings \
    .groupby('arrival_date_year') \
    .arrival_date_month \
    .describe()[['top', 'freq']].loc[[2016, 2017]]
months = bookings \
    .groupby('arrival_date_year') \
    .agg({'arrival_date_month': 'value_counts'}).loc[2016].iloc[0]
# print(months)

# Сгруппируйте данные по годам и проверьте, на какой месяц бронирования отеля
# типа City Hotel отменялись чаще всего в каждый из периодов.
canceled = bookings \
    .query("hotel == 'City Hotel' & is_canceled == 1") \
    .groupby(['arrival_date_year']) \
    .arrival_date_month \
    .describe()[['top', 'freq']]
canceled = bookings \
    .query("hotel == 'City Hotel' & is_canceled == 1") \
    .groupby(['arrival_date_year']) \
    .agg({'arrival_date_month': 'value_counts'}) \
    .groupby("arrival_date_year") \
    .head(1)
# print(canceled)

# Посмотрите на числовые характеристики трёх переменных: adults, children
# и babies. Какая из них имеет наибольшее среднее значение?
peoples = bookings.describe()[['adults', 'children', 'babies']].loc['mean']
peoples = bookings[['adults', 'children', 'babies']].mean()
#print(peoples)

# Создайте колонку total_kids, объединив children и babies. Отели какого типа
# в среднем пользуются большей популярностью у клиентов с детьми?
bookings['total_kids'] = bookings.children + bookings.babies
hotels = bookings.groupby(['hotel']).agg({'total_kids': 'mean'})
hotels = bookings.groupby('hotel').total_kids.mean()
#print(hotels)

# Создайте переменную has_kids, которая принимает значение True, если клиент
# при бронировании указал хотя бы одного ребенка (total_kids), и False –
# в противном случае. Посчитайте отношение количества ушедших пользователей
# к общему количеству клиентов, выраженное в процентах (churn rate). Укажите,
# среди какой группы показатель выше
bookings['has_kids'] = bookings.total_kids > 0
churn_rate = bookings\
    .groupby('has_kids')\
    .agg({'has_kids': 'count', 'is_canceled': 'sum'})\
    .assign(total=lambda x: round(x.is_canceled / x.has_kids * 100, 2))
churn_rate = bookings\
    .groupby('has_kids', as_index=False)\
    .agg({'is_canceled': 'mean'})\
    .round(4)*100
churn_rate = bookings\
    .groupby(['has_kids'])\
    .is_canceled\
    .value_counts(normalize=True)
churn_rate = bookings\
    .pivot_table(index='has_kids',
                 values=['is_canceled'],
                 aggfunc={'has_kids': 'count', 'is_canceled': 'sum'})\
    .rename(columns={'has_kids': 'with_kids'})\
    .reset_index()
churn_rate['total'] = round(churn_rate.is_canceled / churn_rate.with_kids * 100, 2)
print(churn_rate)