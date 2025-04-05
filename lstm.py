import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam

# Load the dataset
file_path = "dataset.csv"
df = pd.read_csv(file_path)

# Convert 'time' column to datetime
df['time'] = pd.to_datetime(df['time'])

# Extract hour, month, and year from the 'time' column
df['hour'] = df['time'].dt.hour
df['month'] = df['time'].dt.month
df['year'] = df['time'].dt.year

# Select relevant features (include additional relevant features if available)
features = ['temperature_2m (°C)', 'relative_humidity_2m (%)', 'precipitation (mm)', 'hour', 'month', 'year']
df = df[['time'] + features]

# Fill missing values with a suitable method, such as interpolation or using a custom imputation method
df.fillna(df.mean(), inplace=True)

# Scale the features
scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])

# Function to create sequences
def create_sequences(data, time_steps=24, forecast_horizon=24):
    X, y = [], []
    for i in range(len(data) - time_steps - forecast_horizon + 1):
        X.append(data[i:i+time_steps])
        y.append(data[i+time_steps + forecast_horizon - 1])
    return np.array(X), np.array(y)

# Prepare the data for training
data = df[features].values
time_steps = 24
forecast_horizon = 24
X, y = create_sequences(data, time_steps, forecast_horizon)

# Split the data into training and testing sets
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Define a more complex model with dropout
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.3),
    LSTM(64, return_sequences=True),
    Dropout(0.3),
    LSTM(64),
    Dense(len(features))
])

# Custom loss function to focus more on precipitation accuracy
def custom_loss(y_true, y_pred):
    mse = tf.keras.losses.MeanSquaredError()
    precipitation_error = mse(y_true[:, 2], y_pred[:, 2])
    total_error = mse(y_true, y_pred)
    return total_error + 2 * precipitation_error

model.compile(optimizer=Adam(learning_rate=0.0005),
              loss=custom_loss,
              metrics=[tf.keras.metrics.RootMeanSquaredError(),
                       tf.keras.metrics.MeanAbsoluteError()])

model.summary()

# Define callbacks
early_stopping_callback = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
    verbose=1
)

checkpoint_callback = ModelCheckpoint(
    "best_weather_prediction_model.h5",
    monitor="val_loss",
    save_best_only=True,
    mode="min",
    verbose=1
)

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=128,
    validation_data=(X_test, y_test),
    verbose=1,
    callbacks=[early_stopping_callback, checkpoint_callback]
)

# Load the best model
best_model = tf.keras.models.load_model("best_weather_prediction_model.h5", custom_objects={'custom_loss': custom_loss})

# Make predictions
predictions = best_model.predict(X_test)
predictions_rescaled = scaler.inverse_transform(predictions)
y_test_rescaled = scaler.inverse_transform(y_test)

# Plot actual vs predicted precipitation
plt.figure(figsize=(12, 5))
plt.plot(y_test_rescaled[:, 2], label="Actual Precipitation")
plt.plot(predictions_rescaled[:, 2], label="Predicted Precipitation")
plt.legend()
plt.show()