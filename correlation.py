import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer

df = pd.read_csv("dataset.csv") 


df_numeric = df.select_dtypes(include=['number']) 


normalizer = Normalizer(norm='l2')
ndf = pd.DataFrame(normalizer.fit_transform(df_numeric), columns=df_numeric.columns)

ndf.drop(columns=['precipitation (mm)', 'snowfall (cm)', 'snow_depth (m)'], inplace=True, errors='ignore')

corr_matrix = ndf.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()
