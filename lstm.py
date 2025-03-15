import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
file_path = "/mnt/data/dataset.csv"
df = pd.read_csv(file_path)
df['time'] = pd.to_datetime(df['time'])
features = ['temperature_2m (°C)', 'relative_humidity_2m (%)', 'precipitation (mm)']
df = df[['time'] + features]
print(df.isnull().sum())
df.fillna(df.mean(), inplace=True)
scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])
def create_sequences(data, time_steps=24):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i+time_steps])  
        y.append(data[i+time_steps])    
    return np.array(X), np.array(y)
data = df[features].values
time_steps = 24
X, y = create_sequences(data, time_steps)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(50),
    Dense(3) 
])
model.compile(optimizer='adam', loss='mse')
model.summary()
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()
predictions = model.predict(X_test)
predictions_rescaled = scaler.inverse_transform(predictions)
y_test_rescaled = scaler.inverse_transform(y_test)
plt.figure(figsize=(12, 5))
plt.plot(y_test_rescaled[:, 0], label="Actual Temperature")
plt.plot(predictions_rescaled[:, 0], label="Predicted Temperature")
plt.legend()
plt.show()

