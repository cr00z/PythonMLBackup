# https://archive.ics.uci.edu/ml/datasets/automobile

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 30)
pd.set_option('display.width', 150)

headers = ["symboling", "normalized_losses", "make", "fuel_type", "aspiration",
           "num_doors", "body_style", "drive_wheels", "engine_location",
           "wheel_base", "length", "width", "height", "curb_weight",
           "engine_type", "num_cylinders", "engine_size", "fuel_system",
           "bore", "stroke", "compression_ratio", "horsepower", "peak_rpm",
           "city_mpg", "highway_mpg", "price"]
df_raw = pd.read_csv('test/imports-85.data.csv',
                     header=None,
                     names=headers,
                     na_values='?')

models = ["toyota", "nissan", "mazda", "honda", "mitsubishi", "subaru",
          "volkswagen", "volvo"]
df = df_raw[df_raw.make.isin(models)].copy()

print(pd.crosstab(df.make, df.body_style))
print(df.groupby(['make', 'body_style'])['body_style'].count().unstack()
        .fillna(0).astype(int))
print(df.pivot_table(index='make',
                     columns='body_style',
                     aggfunc={'body_style': len},
                     fill_value=0))

print(pd.crosstab(df.make, df.body_style, margins=True, margins_name='Total'))
print(pd.crosstab(df.make,
                  df.body_style,
                  values=df.curb_weight,
                  aggfunc='mean')).round(0)
print(pd.crosstab(df.make, df.body_style, normalize=True))
print(pd.crosstab(df.make, df.body_style, normalize='columns'))
print(pd.crosstab(df.make, df.body_style, normalize='index'))
print(pd.crosstab([df.make, df.num_doors],
                  [df.body_style, df.drive_wheels],
                  rownames=['Auto Manufacturer', "Doors"],
                  colnames=['Body Style', "Drive Type"],
                  dropna=False))
table = pd.crosstab(df.make,
                    [df.body_style, df.drive_wheels])
print(table)
sns.heatmap(table, cmap='YlGnBu', annot=True, cbar=False)
plt.show()