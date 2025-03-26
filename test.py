import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

model = tf.keras.models.load_model("best_weather_prediction_model.h5")
print("✅ Model loaded successfully!")

file_path = "dataset.csv"
df = pd.read_csv(file_path)

df['time'] = pd.to_datetime(df['time'])
df['hour'] = df['time'].dt.hour
df['month'] = df['time'].dt.month
df['year'] = df['time'].dt.year

features = ['temperature_2m (°C)', 'relative_humidity_2m (%)', 'precipitation (mm)', 'hour', 'month', 'year']
df = df[['time'] + features]

df.fillna(df.mean(), inplace=True)

scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])

specific_time = pd.to_datetime("2009-09-22 15:00:00")
past_24_hours = df[df['time'] < specific_time].tail(24)

if len(past_24_hours) < 24:
    raise ValueError("Not enough past data for prediction!")

past_24_hours_features = past_24_hours[features].values
past_24_hours_features = past_24_hours_features.reshape((1, past_24_hours_features.shape[0], past_24_hours_features.shape[1]))


predicted_scaled = model.predict(past_24_hours_features)
predicted_weather = scaler.inverse_transform(predicted_scaled)


actual_weather = df[df['time'] == specific_time][features].values

if actual_weather.shape[0] == 0:
    raise ValueError("No actual data available for comparison!")

actual_weather = scaler.inverse_transform(actual_weather)


mae = mean_absolute_error(actual_weather[0], predicted_weather[0])
rmse = np.sqrt(mean_squared_error(actual_weather[0], predicted_weather[0]))


print(f"Predicted Weather for {specific_time}:")
print(f"Temperature: {predicted_weather[0][0]:.2f}°C")
print(f"Humidity: {predicted_weather[0][1]:.2f}%")
print(f"Precipitation: {predicted_weather[0][2]:.2f} mm")

print("\nError Metrics:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
