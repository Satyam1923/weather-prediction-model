import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler


df = pd.read_csv('dataset.csv', parse_dates=['time'])


df['hour'] = df['time'].dt.hour
df['day'] = df['time'].dt.day
df['month'] = df['time'].dt.month
df['year'] = df['time'].dt.year

y = df['weather_code (wmo code)'].astype(int)

X = df.select_dtypes(include=['number']).drop(columns=['weather_code (wmo code)', 'rain (mm)', 'snowfall (cm)', 'snow_depth (m)'])

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

mi_scores = mutual_info_classif(X_scaled, y)
mi_scores = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)

print("\nMutual Information Scores:\n", mi_scores)

top_features = mi_scores[mi_scores > 0.01].index.tolist()
print("\nSelected Features:", top_features)

X_selected = X_scaled[top_features]

X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)

log_reg_model = LogisticRegression(max_iter=1000, random_state=42)

rf_model.fit(X_train, y_train)
log_reg_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
y_pred_log_reg = log_reg_model.predict(X_test)

X_test_copy = X_test.copy()
X_test_copy['Random Forest Prediction'] = y_pred_rf
X_test_copy['Logistic Regression Prediction'] = y_pred_log_reg
X_test_copy.to_csv("predictions.csv", index=False)

print("Predictions saved to predictions.csv!")

accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("\nRandom Forest Accuracy:", accuracy_rf)
print("\nRandom Forest Classification Report:\n", classification_report(y_test, y_pred_rf))

accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)
print("\nLogistic Regression Accuracy:", accuracy_log_reg)
print("\nLogistic Regression Classification Report:\n", classification_report(y_test, y_pred_log_reg))

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt="d", cmap="Blues",
            xticklabels=np.unique(y), yticklabels=np.unique(y), ax=axes[0])
axes[0].set_title("Random Forest - Confusion Matrix")
axes[0].set_xlabel("Predicted Label")
axes[0].set_ylabel("True Label")

sns.heatmap(confusion_matrix(y_test, y_pred_log_reg), annot=True, fmt="d", cmap="Blues",
            xticklabels=np.unique(y), yticklabels=np.unique(y), ax=axes[1])
axes[1].set_title("Logistic Regression - Confusion Matrix")
axes[1].set_xlabel("Predicted Label")
axes[1].set_ylabel("True Label")

plt.show()
