import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('dataset.csv', parse_dates=['time'])

df['hour'] = df['time'].dt.hour
df['day'] = df['time'].dt.day
df['month'] = df['time'].dt.month
df['year'] = df['time'].dt.year
df['dayofweek'] = df['time'].dt.dayofweek
df['is_weekend'] = df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)

y = df['weather_code (wmo code)'].astype(int)  

X = df.select_dtypes(include=['number']).drop(columns=['weather_code (wmo code)', 'rain (mm)', 'cloud_cover (%)'])

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

mi_scores = mutual_info_classif(X_scaled, y)
mi_scores = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)


print("\nMutual Information Scores:\n", mi_scores)

top_features = mi_scores[mi_scores > 0.01].index.tolist()  
print("\nSelected Features:", top_features)

X_selected = X_scaled[top_features]

X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)

rf_model = RandomForestClassifier(
    n_estimators=200, 
    max_depth=10,  
    min_samples_split=10, 
    min_samples_leaf=5, 
    random_state=42
)

rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy Score:", accuracy)


print("\nClassification Report:\n", classification_report(y_test, y_pred))

plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
