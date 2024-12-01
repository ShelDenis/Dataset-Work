import pandas as pd
import numpy as np
import seaborn as sb
import scipy as sp
import matplotlib.pyplot as plt
import random


def get_random_color():
    c = []
    for j in range(3):
        c.append(random.randint(1, 255) / 255)
    return c


df = pd.read_csv('data/world_cities_data.csv', delimiter=',')
print(df[:5])
print(df[-5:])
print(df.info())
print(df.describe())
df = df.drop_duplicates()

df = df.drop(columns=['id'])

df.rename(columns={'rank': 'rating'}, inplace=True)

lats = df['lat']
plt.xlim([-90, 90])
plt.ylim([0, 300])
plt.title('')
plt.xlabel('Широта')
plt.ylabel('Количество городов')
plt.hist(lats, 1000, color=get_random_color())
plt.savefig('data/latitude_hist.png')
plt.clf()

longs = df['lng']
plt.boxplot(x=longs)
plt.savefig('data/longitude_boxplot.png')
plt.clf()

uniq, counts = np.unique(df['country'], return_counts=True)
uniq_tuple = [(uniq[i], counts[i]) for i in range(len(uniq))]
uniq_tuple.sort(key=lambda x:-x[1])
cut_number = 15
extra_number = len(uniq_tuple) - cut_number
uniq_tuple = uniq_tuple[:cut_number]
counts = [x[1] for x in uniq_tuple]
counts.append(extra_number)
uniq = [x[0] for x in uniq_tuple]
uniq.append('Other')
plt.pie(counts, labels=[x for x in uniq])
plt.savefig('data/cities_number_in_countries.png')
plt.clf()

print(df.corr(numeric_only=True))
dataplot = sb.heatmap(df.corr(numeric_only=True), cmap="YlGnBu", annot=True)
plt.savefig('data/correlation.png')
plt.clf()

cut_df = df[:7]
sb.barplot(x="city", y="population", data = cut_df)
plt.savefig('data/countplot.png')
plt.clf()


keys = df.columns.values
cols_with_empties = []
for k in keys:
    if any(df[k].isnull()):
        print(f"Столбец {k} имеет пустые ячейки")
        cols_with_empties.append(k)

print('Было:')
print(df[-5:])

for k in cols_with_empties:
    filling = 0
    if df[k].dtype == int:
        filling = sp.ndimage.median(df[k])
    elif df[k].dtype == float:
        filling = np.mean(df[k])
    else:
        filling = sp.stats.mode(df[k])
    df[k] = df[k].fillna(filling)

print('\nСтало:')
print(df[-5:])


df_part = df[:200]
ntest = sp.stats.normaltest(df_part['population'])
if ntest[1] < 0.05:
    print('Распределение не является нормальным')
else:
    print('Распределение нормальное')


df_encoded = pd.get_dummies(df)
df_encoded.to_csv('data/cities_stats.csv', sep='\t')