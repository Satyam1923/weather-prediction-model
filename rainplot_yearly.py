import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('dataset.csv')

df['time'] = pd.to_datetime(df['time'])

year = 2008
df_year = df[(df['time'].dt.year == year) & (df['rain (mm)'] > 0)]

plt.figure(figsize=(10, 5))
plt.plot(df_year['time'], df_year['rain (mm)'], marker='o', linestyle='None', markersize=3)

plt.xlabel("Time")
plt.ylabel("Rainfall (mm)")
plt.title(f"Nonzero Rainfall Data Over {year}")

plt.xticks(rotation=45)

plt.show()
