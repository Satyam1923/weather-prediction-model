import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def custom_loss(y_true, y_pred):
    mse = tf.keras.losses.MeanSquaredError()
    precipitation_error = mse(y_true[:, 2], y_pred[:, 2])
    total_error = mse(y_true, y_pred)
    return total_error + 2 * precipitation_error

# Load the model
model = tf.keras.models.load_model("best_weather_prediction_model.h5", custom_objects={'custom_loss': custom_loss})
print("✅ Model loaded successfully!")

# Load the dataset
file_path = "dataset.csv"
df = pd.read_csv(file_path)

# Convert 'time' column to datetime
df['time'] = pd.to_datetime(df['time'])

# Extract hour, month, and year from the 'time' column
df['hour'] = df['time'].dt.hour
df['month'] = df['time'].dt.month
df['year'] = df['time'].dt.year

# Select relevant features
features = ['temperature_2m (°C)', 'relative_humidity_2m (%)', 'precipitation (mm)', 'hour', 'month', 'year']
df = df[['time'] + features]

# Fill missing values with the mean
df.fillna(df.mean(), inplace=True)

# Scale the features
scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])

# Define the start time for the 7-day prediction
start_time = pd.to_datetime("2007-9-22 16:00:00")

# Initialize lists to store predictions and actual values
predicted_temperatures = []
predicted_humidities = []
predicted_rain = []
actual_temperatures = []
actual_humidities = []
actual_rain = []

# Function to create input sequence
def create_input_sequence(df, start_time, past_steps=24):
    end_time = start_time - pd.DateOffset(hours=1)
    past_data = df[df['time'] <= end_time].tail(past_steps)[features].values
    return past_data.reshape((1, past_data.shape[0], past_data.shape[1]))

# Iterate over 7 days
dates = []
for day in range(7):
    specific_time = start_time + pd.DateOffset(days=day)
    dates.append(specific_time)
    
    # Get the past 24 hours of data before the specific time
    past_24_hours_features = create_input_sequence(df, specific_time)

    # Make the prediction
    if past_24_hours_features.shape[1] == 24:
        predicted_scaled = model.predict(past_24_hours_features)
        predicted_weather = scaler.inverse_transform(predicted_scaled)
        
        # Store the results
        predicted_temperatures.append(predicted_weather[0][0])
        predicted_humidities.append(predicted_weather[0][1])
        predicted_rain.append("Yes" if predicted_weather[0][2] > 0 else "No")

        # Get the actual weather data for the specific day (use mean over 24 hours)
        actual_day_data = df[(df['time'] >= specific_time) & (df['time'] < specific_time + pd.DateOffset(days=1))][features]
        if not actual_day_data.empty:
            actual_weather = scaler.inverse_transform(actual_day_data.mean().values.reshape(1, -1))
            actual_temperatures.append(actual_weather[0][0])
            actual_humidities.append(actual_weather[0][1])
            actual_rain.append("Yes" if actual_weather[0][2] > 0 else "No")
        else:
            actual_temperatures.append(np.nan)
            actual_humidities.append(np.nan)
            actual_rain.append(np.nan)
    else:
        print(f"Not enough past data for prediction on day {day+1}! Skipping...")

# Plot actual vs predicted values for temperature, humidity, and rain over 7 days
plt.figure(figsize=(16, 10))

# Temperature plot
plt.subplot(2, 2, 1)
plt.plot(dates, predicted_temperatures, label='Predicted Temperature', marker='o', color='orange')
plt.plot(dates, actual_temperatures, label='Actual Temperature', marker='o', color='blue')
plt.title('Temperature (°C) over 7 Days')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.xticks(rotation=45, ha='right')
plt.legend()

# Humidity plot
plt.subplot(2, 2, 2)
plt.plot(dates, predicted_humidities, label='Predicted Humidity', marker='o', color='orange')
plt.plot(dates, actual_humidities, label='Actual Humidity', marker='o', color='blue')
plt.title('Humidity (%) over 7 Days')
plt.xlabel('Date')
plt.ylabel('Humidity')
plt.xticks(rotation=45, ha='right')
plt.legend()

# Rain plot
predicted_rain_markers = [1 if pred == "Yes" else 0 for pred in predicted_rain]
actual_rain_markers = [1 if act == "Yes" else 0 for act in actual_rain]

plt.subplot(2, 2, 3)
plt.plot(dates, predicted_rain_markers, label='Predicted Rain', marker='o', color='green')
plt.plot(dates, actual_rain_markers, label='Actual Rain', marker='o', color='red')
plt.title('Rain Predictions over 7 Days')
plt.xlabel('Date')
plt.ylabel('Rain')
plt.yticks([0, 1], ['No', 'Yes'])
plt.xticks(rotation=45, ha='right')
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()

# Print rain predictions
print("\nRain Predictions over 7 Days:")
for day, date in enumerate(dates):
    print(f"{date.strftime('%Y-%m-%d')} - Predicted: {predicted_rain[day]}, Actual: {actual_rain[day]}")