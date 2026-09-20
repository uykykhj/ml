import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

df = pd.read_csv("car_prices.csv")

t1=df.head(10)
t2=df.isnull().sum()

df['make'] = df['make'].fillna(df['make'].mode()[0])
df['model'] = df['model'].fillna(df['model'].mode()[0])
df['trim'] = df['trim'].fillna(df['trim'].mode()[0])
df['body'] = df['body'].fillna(df['body'].mode()[0])
df['transmission'] = df['transmission'].fillna(df['transmission'].mode()[0])
df['color'] = df['color'].fillna(df['color'].mode()[0])
df['interior'] = df['interior'].fillna(df['interior'].mode()[0])
df['condition'] = df['condition'].fillna(df['condition'].median())
df['odometer'] = df['odometer'].fillna(df['odometer'].median())
df['mmr'] = df['mmr'].fillna(df['mmr'].median())
df['sellingprice'] = df['sellingprice'].fillna(df['sellingprice'].median())
df['saledate'] = df['saledate'].fillna(df['saledate'].mode()[0])

print(df.isnull().sum())



n_c = ['condition', 'odometer', 'mmr']
scaler = MinMaxScaler()
df[n_c] = scaler.fit_transform(df[n_c])

scaler_std = StandardScaler()
df['year'] = scaler_std.fit_transform(df[['year']])


df = pd.get_dummies(df, columns=['make', 'body', 'transmission', 'color', 'interior', 'state', 'seller'], drop_first=True)
df = df.drop(['model', 'trim', 'vin', 'saledate'], axis='columns')

