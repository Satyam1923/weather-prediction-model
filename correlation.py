import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("dataset.csv")  

date_columns = ["time"]  
for col in date_columns:
    df[col] = pd.to_datetime(df[col], format="%Y-%m-%d", errors="coerce")

df_numeric = df.select_dtypes(include=['number'])


corr_matrix = df_numeric.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.show()
