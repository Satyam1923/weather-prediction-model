import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split


df = pd.read_csv('dataset.csv')

X = df.drop(columns=['rain (mm)','time','precipitation (mm)','weather_code (wmo code)'])
y = df['rain (mm)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators=100, random_state=42)

rf.fit(X_train, y_train)

feature_importances = rf.feature_importances_

importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

print(importance_df)
selector = SelectFromModel(rf, threshold='median', prefit=True)
X_selected = selector.transform(X)

selected_features = X.columns[selector.get_support()]
print("Selected Features:", selected_features)

from sklearn.metrics import r2_score, mean_squared_error

X_train_selected, X_test_selected, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

rf_selected = RandomForestRegressor(n_estimators=100, random_state=42)
rf_selected.fit(X_train_selected, y_train)

y_pred = rf_selected.predict(X_test_selected)


r2 = r2_score(y_test, y_pred)
print("R² Score:", r2)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)


import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='b')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--r')  
plt.xlabel("Actual Rainfall")
plt.ylabel("Predicted Rainfall")
plt.title("Actual vs. Predicted Rainfall")
plt.show()