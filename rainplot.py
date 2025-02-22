import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('dataset.csv')

df['time'] = pd.to_datetime(df['time'])

df_nonzero = df[df['rain (mm)'] > 0]


plt.figure(figsize=(12, 5))
plt.plot(df_nonzero['time'], df_nonzero['rain (mm)'], marker='o', linestyle='None', markersize=3)

plt.xlabel("Time")
plt.ylabel("Rainfall (mm)")
plt.title("Nonzero Rainfall Data Over Time")

plt.xticks(rotation=45)

plt.show()
