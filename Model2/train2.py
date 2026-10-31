import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error # Import mean_absolute_error
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
    Dense(64, activation='relu', input_shape=(5,)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
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

# Step 8: Calculate MSE
mse = mean_squared_error(y_actual, y_pred)
print("Mean Squared Error (MSE):", mse)
mae = mean_absolute_error(y_actual, y_pred)
var = np.var(y_actual)
relative_mse = mse / var
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

# # # --- Plot  MSE vs Relative MSE ---
# plt.figure(figsize=(6, 4))
# plt.bar(['MSE', 'Relative MSE'], [mse, relative_mse], color=['skyblue', 'lightgreen'])
# plt.title('MSE vs Relative MSE')
# plt.ylabel('Error')
# plt.grid(True, axis='y')
# plt.tight_layout()
# plt.show()

import tensorflow as tf

# 1. Convert trained Model 2 to float32 TFLite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# 2. Helper to format numpy arrays as C float arrays
def to_c_float_array(arr, var_name):
    values = ", ".join([f"{val:.8f}f" for val in arr.flatten()])
    return f"const float {var_name}[{len(arr.flatten())}] = {{{values}}};\n"

# 3. Generate model2_data.h in your script's directory
header_path = Path(__file__).with_name("model2_data.h")

with open(header_path, "w") as f:
    f.write("#ifndef MODEL2_DATA_H\n#define MODEL2_DATA_H\n\n")
    
    # Write StandardScaler parameters for X (5 inputs) and y (7 outputs)
    f.write("// StandardScaler constants for Model 2\n")
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
    f.write("\n};\n\n#endif // MODEL2_DATA_H\n")

print(f"Successfully generated {header_path}")

import time
import serial
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# --- CONFIGURATION ---
SERIAL_PORT = "COM6"   # <-- Change to your ESP32 COM port
BAUD_RATE = 115200
MAX_TEST_SAMPLES = 200 # Set to len(X_test) to stream the full test set

num_samples = min(MAX_TEST_SAMPLES, len(X_test))
X_test_subset = X_test[:num_samples]
y_actual_subset = y_actual[:num_samples]

print(f"Connecting to ESP32 (Model 2) on {SERIAL_PORT} at {BAUD_RATE} baud...")
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
time.sleep(2.0)
ser.reset_input_buffer()

y_pred_esp32 = []

print(f"Streaming {num_samples} test samples to ESP32 for Model 2 inference...")
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

# --- Calculate Model 2 ESP32 Hardware Metrics ---
esp_mse = mean_squared_error(y_actual_subset, y_pred_esp32)
esp_mae = mean_absolute_error(y_actual_subset, y_pred_esp32)
esp_var = np.var(y_actual_subset)
esp_relative_mse = esp_mse / esp_var

print("\n================ ESP32 HARDWARE TEST RESULTS ================")
print("ESP32 Mean Squared Error (MSE):", esp_mse)
print(f"ESP32 Mean Absolute Error (MAE): {esp_mae:.6f}")
print(f"ESP32 Relative MSE: {esp_relative_mse:.6f}")
print("=============================================================\n")

# --- Graph 1: Loss (MSE) Plot ---
plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='Training loss', color='yellow')
plt.plot(history.history['val_loss'], label='Validation loss', color='blue')
plt.axhline(y=np.mean((y_test_scaled[:num_samples] - scaler_y.transform(y_pred_esp32))**2),
            color='cyan', linestyle='--', label='ESP32 Hardware Test Loss (Scaled)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Model 2: Training and validation loss (with ESP32 Hardware Test Benchmark)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Graph 2: Mean Absolute Error (MAE) Plot ---
plt.figure(figsize=(8, 4))
plt.plot(history.history['mae'], label='Training Mean absolute error', color='green')
plt.plot(history.history['val_mae'], label='Validation Mean absolute error', color='red')
plt.axhline(y=np.mean(np.abs(y_test_scaled[:num_samples] - scaler_y.transform(y_pred_esp32))),
            color='orange', linestyle='--', label='ESP32 Hardware Test MAE (Scaled)')
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error')
plt.title('Model 2: Training and validation Mean absolute error (with ESP32 Benchmark)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Graph 3: ESP32 Hardware MSE vs Relative MSE Bar Chart ---
plt.figure(figsize=(6, 4))
plt.bar(['MSE', 'Relative MSE'], [esp_mse, esp_relative_mse], color=['skyblue', 'lightgreen'])
plt.title('Model 2 ESP32 Hardware: MSE vs Relative MSE')
plt.ylabel('Error')
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()

y_pred_esp32 = np.array(y_pred_esp32)
y_pred_python_subset = y_pred[:num_samples]

# --- 1. Compute Unscaled Metrics for Model 2 (Python vs Arduino ESP32 on the same test subset) ---
py_mse = mean_squared_error(y_actual_subset, y_pred_python_subset)
py_mae = mean_absolute_error(y_actual_subset, y_pred_python_subset)
py_var = np.var(y_actual_subset)
py_relative_mse = py_mse / py_var

esp_mse = mean_squared_error(y_actual_subset, y_pred_esp32)
esp_mae = mean_absolute_error(y_actual_subset, y_pred_esp32)
esp_var = np.var(y_actual_subset)
esp_relative_mse = esp_mse / esp_var

# --- 2. Compute Scaled Metrics for Model 2 (to match Keras Training/Validation scale on Plots 1 & 2) ---
y_pred_esp32_scaled = scaler_y.transform(y_pred_esp32)

py_test_loss_scaled = np.mean((y_test_scaled[:num_samples] - y_pred_scaled[:num_samples]) ** 2)
esp_test_loss_scaled = np.mean((y_test_scaled[:num_samples] - y_pred_esp32_scaled) ** 2)

py_test_mae_scaled = np.mean(np.abs(y_test_scaled[:num_samples] - y_pred_scaled[:num_samples]))
esp_test_mae_scaled = np.mean(np.abs(y_test_scaled[:num_samples] - y_pred_esp32_scaled))

print("\n============ MODEL 2: MERGED PYTHON vs ARDUINO (ESP32) METRICS ===========")
print(f"Metric         | Python (PC)   | Arduino (ESP32) | Absolute Diff")
print("--------------------------------------------------------------------------")
print(f"MSE            | {py_mse:13.6f} | {esp_mse:15.6f} | {abs(py_mse - esp_mse):.8f}")
print(f"MAE            | {py_mae:13.6f} | {esp_mae:15.6f} | {abs(py_mae - esp_mae):.8f}")
print(f"Relative MSE   | {py_relative_mse:13.6f} | {esp_relative_mse:15.6f} | {abs(py_relative_mse - esp_relative_mse):.8f}")
print("==========================================================================\n")

# # --- Graph 1: Model 2 Merged Loss (MSE) Plot (Python + ESP32) ---
# plt.figure(figsize=(8, 4))
# plt.plot(history.history['loss'], label='Training loss', color='yellow')
# plt.plot(history.history['val_loss'], label='Validation loss', color='blue')
# plt.axhline(y=py_test_loss_scaled, color='limegreen', linestyle='-',
#             label=f'Python Test Loss ({py_test_loss_scaled:.4f})')
# plt.axhline(y=esp_test_loss_scaled, color='cyan', linestyle='--',
#             label=f'ESP32 Hardware Test Loss ({esp_test_loss_scaled:.4f})')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.title('Model 2: Training and validation loss (Python vs ESP32 Benchmark)')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# # --- Graph 2: Model 2 Merged Mean Absolute Error (MAE) Plot (Python + ESP32) ---
# plt.figure(figsize=(8, 4))
# plt.plot(history.history['mae'], label='Training Mean absolute error', color='green')
# plt.plot(history.history['val_mae'], label='Validation Mean absolute error', color='red')
# plt.axhline(y=py_test_mae_scaled, color='dodgerblue', linestyle='-',
#             label=f'Python Test MAE ({py_test_mae_scaled:.4f})')
# plt.axhline(y=esp_test_mae_scaled, color='orange', linestyle='--',
#             label=f'ESP32 Hardware Test MAE ({esp_test_mae_scaled:.4f})')
# plt.xlabel('Epochs')
# plt.ylabel('Mean Absolute Error')
# plt.title('Model 2: Training and validation Mean absolute error (Python vs ESP32 Benchmark)')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# # --- Graph 3: Model 2 Merged MSE vs Relative MSE Bar Chart (Python + ESP32) ---
# plt.figure(figsize=(8, 4))
# plt.bar(
#     ['Python MSE', 'Python Relative MSE', 'ESP32 MSE', 'ESP32 Relative MSE'],
#     [py_mse, py_relative_mse, esp_mse, esp_relative_mse],
#     color=['skyblue', 'lightgreen', 'skyblue', 'lightgreen']
# )
# plt.title('Model 2: MSE vs Relative MSE (Python vs ESP32 Hardware)')
# plt.ylabel('Error')
# plt.grid(True, axis='y')
# plt.tight_layout()
# plt.show()