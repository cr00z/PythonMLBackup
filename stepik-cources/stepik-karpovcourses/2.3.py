import pandas as pd

pd.set_option('display.width', 320)
pd.set_option('display.max_columns', 20)


def temp_to_celcius(fahr):
    return (fahr - 32) * 5 / 9
    #return [(f - 32) * 5 / 9 for f in fahr]
    #return fahr.map(lambda f: (f - 32) * 5 / 9)


taxi = pd.read_csv('test/2_taxi_nyc.csv')
print(taxi.head())
taxi.columns = taxi.columns.str.replace(' ', '_')
taxi.rename(columns=lambda x: x.replace(' ', '_'), inplace=True)
print(taxi.query("borough == 'Brooklyn'").shape[0])
print(taxi.pickups.sum())
print(taxi.groupby(['borough'])
      .agg({'pickups': 'sum'})
      .idxmax())
min_pickups = taxi.groupby(['borough'])\
    .agg({'pickups': 'sum'})\
    .idxmin()
print(min_pickups)
holiday_trip = taxi\
    .groupby(['borough', 'hday'])\
    .agg({'pickups': 'mean'})
holiday_trip = taxi\
    .pivot_table(index='borough', columns='hday', values='pickups', aggfunc='mean')\
    .query('Y > N')
print(holiday_trip)
pickups_by_mon_bor = taxi \
    .groupby(['pickup_month', 'borough'], as_index=False) \
    .agg({'pickups': 'sum'}) \
    .sort_values(['pickups'], ascending=False)
print(pickups_by_mon_bor)
taxi['temp_C'] = temp_to_celcius(taxi['temp'])
print(taxi['temp_C'][:5])
print(temp_to_celcius(taxi.loc[:149].temp.values))