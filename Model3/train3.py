import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score # Import mean_absolute_error and r2_score
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping

#  Load dataset
df = pd.read_excel(Path(__file__).with_name("dataCombined1.csv"))

# Handle missing values by dropping rows with NaNs
df = df.dropna()

#  Define input and output
X = df[['frequency','Rimp', 'Ximp', 'Directivity', 'S11_dB']].values  # Last 5 columns
y = df.iloc[:, :7].values  # First 7 columns

#  Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize input and output
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

#  Define the model
model = Sequential([
    Dense(64, activation='sigmoid', input_shape=(5,)),
    Dense(128, activation='tanh'),
    Dense(64, activation='tanh'),
    Dense(7)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(X_train_scaled, y_train_scaled, validation_split=0.2, epochs=100, batch_size=64, callbacks=[early_stop], verbose=1)

#  Predict and inverse scale
y_pred_scaled = model.predict(X_test_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_actual = scaler_y.inverse_transform(y_test_scaled)

# Predict on training data for R2 score
y_train_pred_scaled = model.predict(X_train_scaled)
y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled)

# Step 8: Calculate MSE
mse = mean_squared_error(y_actual, y_pred)
print("Mean Squared Error (MSE):", mse)
mae = mean_absolute_error(y_actual, y_pred)
var = np.var(y_actual)
relative_mse = mse / var
r2_train = r2_score(y_train, y_train_pred) # Corrected R2 calculation
r2_test = r2_score(y_test, y_pred)         # Corrected R2 calculation
print(f"R-squared (R2) on training set: {r2_train:.6f}")
print(f"R-squared (R2) on test set: {r2_test:.6f}")
print(f"Mean Absolute Error (MAE): {mae:.6f}")
print(f"Relative MSE: {relative_mse:.6f}")


# # Plot: Training vs Validation Loss (MSE)
# plt.figure(figsize=(8, 4))
# plt.plot(history.history['loss'], label='Training loss', color='yellow')
# plt.plot(history.history['val_loss'], label='Validation loss', color='blue')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.title('Training and validation loss')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# # Plot: MAE
# plt.figure(figsize=(8, 4))
# plt.plot(history.history['mae'], label='Training Mean absolute error', color='green')
# plt.plot(history.history['val_mae'], label='Validation Mean absolute error', color='red')
# plt.xlabel('Epochs')
# plt.ylabel('Mean Absolute Error')
# plt.title('Training and validation Mean absolute error')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

import tensorflow as tf

# 1. Convert trained Model 3 to float32 TFLite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# 2. Helper to format numpy arrays as C float arrays
def to_c_float_array(arr, var_name):
    values = ", ".join([f"{val:.8f}f" for val in arr.flatten()])
    return f"const float {var_name}[{len(arr.flatten())}] = {{{values}}};\n"

# 3. Generate model3_data.h in your script's directory
header_path = Path(__file__).with_name("model3_data.h")

with open(header_path, "w") as f:
    f.write("#ifndef MODEL3_DATA_H\n#define MODEL3_DATA_H\n\n")
    
    # Write StandardScaler parameters for X (5 inputs) and y (7 outputs)
    f.write("// StandardScaler constants for Model 3\n")
    f.write(to_c_float_array(scaler_X.mean_, "scaler_X_mean"))
    f.write(to_c_float_array(scaler_X.scale_, "scaler_X_scale"))
    f.write(to_c_float_array(scaler_y.mean_, "scaler_y_mean"))
    f.write(to_c_float_array(scaler_y.scale_, "scaler_y_scale"))
    f.write("\n")
    
    # Write reference test sample 0 for initial ESP32 verification
    f.write("// Reference sample 0 from X_test and y_actual\n")
    f.write(to_c_float_array(X_test[0], "test_input_sample"))
    f.write(to_c_float_array(y_actual[0], "expected_output_sample"))
    f.write(to_c_float_array(y_pred[0], "python_predicted_sample"))
    f.write("\n")

    # Write the TFLite model byte array
    f.write(f"const unsigned int model_data_len = {len(tflite_model)};\n")
    f.write("alignas(8) const unsigned char model_data[] = {\n  ")
    hex_bytes = [f"0x{b:02x}" for b in tflite_model]
    for i in range(0, len(hex_bytes), 12):
        f.write(", ".join(hex_bytes[i:i+12]) + ",\n  ")
    f.write("\n};\n\n#endif // MODEL3_DATA_H\n")

print(f"Successfully generated {header_path}")

import time
import serial
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- CONFIGURATION ---
SERIAL_PORT = "COM6"   # <-- Change to your ESP32 COM port
BAUD_RATE = 115200
MAX_TEST_SAMPLES = 200 # Set to len(X_test) to stream the full test set

num_samples = min(MAX_TEST_SAMPLES, len(X_test))
X_test_subset = X_test[:num_samples]
y_actual_subset = y_actual[:num_samples]

print(f"Connecting to ESP32 (Model 3) on {SERIAL_PORT} at {BAUD_RATE} baud...")
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
time.sleep(2.0)
ser.reset_input_buffer()

y_pred_esp32 = []

print(f"Streaming {num_samples} test samples to ESP32 for Model 3 inference...")
for idx, row in enumerate(X_test_subset):
    payload = ",".join([f"{val:.6f}" for val in row]) + "\n"
    ser.write(payload.encode("utf-8"))

    response = ser.readline().decode("utf-8", errors="ignore").strip()
    parts = response.split(",")
    if len(parts) == 7:
        esp_row = [float(x) for x in parts]
    else:
        print(f"Warning on sample {idx}: received '{response}', falling back to zeros")
        esp_row = [0.0] * 7

    y_pred_esp32.append(esp_row)

    if (idx + 1) % 25 == 0 or (idx + 1) == num_samples:
        print(f"Processed {idx + 1}/{num_samples} samples on ESP32...")

ser.close()
y_pred_esp32 = np.array(y_pred_esp32)

# --- Calculate Model 3 ESP32 Hardware Metrics ---
esp_mse = mean_squared_error(y_actual_subset, y_pred_esp32)
esp_mae = mean_absolute_error(y_actual_subset, y_pred_esp32)
esp_var = np.var(y_actual_subset)
esp_relative_mse = esp_mse / esp_var
esp_r2_test = r2_score(y_actual_subset, y_pred_esp32)

print("ESP32 Mean Squared Error (MSE):", esp_mse)
print(f"ESP32 R-squared (R2) on test set: {esp_r2_test:.6f}")
print(f"ESP32 Mean Absolute Error (MAE): {esp_mae:.6f}")
print(f"ESP32 Relative MSE: {esp_relative_mse:.6f}")

# --- Graph 1: Training vs Validation Loss (MSE) with ESP32 Benchmark ---
plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='Training loss', color='yellow')
plt.plot(history.history['val_loss'], label='Validation loss', color='blue')
plt.axhline(y=np.mean((y_test_scaled[:num_samples] - scaler_y.transform(y_pred_esp32))**2),
            color='cyan', linestyle='--', label='ESP32 Hardware Test Loss (Scaled)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Model 3: Training and validation loss (with ESP32 Hardware Test Benchmark)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Graph 2: MAE with ESP32 Benchmark ---
plt.figure(figsize=(8, 4))
plt.plot(history.history['mae'], label='Training Mean absolute error', color='green')
plt.plot(history.history['val_mae'], label='Validation Mean absolute error', color='red')
plt.axhline(y=np.mean(np.abs(y_test_scaled[:num_samples] - scaler_y.transform(y_pred_esp32))),
            color='orange', linestyle='--', label='ESP32 Hardware Test MAE (Scaled)')
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error')
plt.title('Model 3: Training and validation Mean absolute error (with ESP32 Benchmark)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()