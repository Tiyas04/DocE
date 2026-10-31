# TinyML Multi-Output Regression on ESP32: Antenna Parameter Modeling & Hardware-in-the-Loop Benchmarking

## 1. Project Overview
This project implements an end-to-end **TinyML deployment and hardware-in-the-loop (HIL) verification pipeline** for multi-output regression neural networks on the **ESP32 microcontroller** using **TensorFlow Lite for Microcontrollers (TFLite Micro)**. 

The models take **5 electromagnetic/antenna performance parameters** as inputs and predict **7 corresponding physical/geometric design parameters** as outputs. To ensure zero accuracy degradation on the edge device, the models are deployed in 32-bit floating-point (`float32`) format alongside embedded `StandardScaler` normalization constants, and verified over USB Serial against desktop Python predictions.

---

## 2. Dataset & Feature Mapping
* **Dataset File:** `dataCombined1.csv`
* **Preprocessing:** Rows with missing values (`NaN`) are dropped (`df.dropna()`), and the dataset is split into **80% training** and **20% testing** (`random_state=42`).
* **Input Features ($X \in \mathbb{R}^5$):**
  1. `frequency`
  2. `Rimp` (Input Resistance)
  3. `Ximp` (Input Reactance)
  4. `Directivity`
  5. `S11_dB` (Return Loss in dB)
* **Target Outputs ($y \in \mathbb{R}^7$):**
  * First 7 columns of the dataset (`df.iloc[:, :7]`), representing the target geometric/structural parameters.
* **Feature Scaling:** Both inputs ($X$) and outputs ($y$) are standardized to zero mean and unit variance using `scikit-learn`'s `StandardScaler` (`scaler_X` and `scaler_y`).

---

## 3. Neural Network Architectures Evaluated

All three models share the same topology (`5 -> 64 -> 128 -> 64 -> 7`, totaling **17,415 trainable parameters** or ~70 KB in `float32`), trained with the **Adam** optimizer, **Mean Squared Error (`mse`)** loss, **Mean Absolute Error (`mae`)** metric, `batch_size=64`, `epochs=100`, and `EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)`.

| Model | Layer 1 (64 units) | Layer 2 (128 units) | Layer 3 (64 units) | Output Layer (7 units) | TFLite Micro Op Resolver Requirements |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Model 1** | `Dense` + `ReLU` | `Dense` + `ReLU` | `Dense` + `ReLU` | `Dense` (Linear) | `AddFullyConnected()`, `AddRelu()` |
| **Model 2** | `Dense` + `ReLU` | `Dense` + `ReLU` | `Dense` + `ReLU` | `Dense` (Linear) | `AddFullyConnected()`, `AddRelu()` |
| **Model 3** | `Dense` + `Sigmoid` | `Dense` + `Tanh` | `Dense` + `Tanh` | `Dense` (Linear) | `AddFullyConnected()`, `AddLogistic()`, `AddTanh()` |

> **Note on Model 3 Activations:** In TFLite Micro, Keras `activation='sigmoid'` maps to the `LOGISTIC` operator (`resolver.AddLogistic()`), and `activation='tanh'` maps to the `TANH` operator (`resolver.AddTanh()`).

---

## 4. End-to-End TinyML Deployment Workflow

```
+-------------------------------------------------------------------+
| 1. PC (Python / Keras / Scikit-Learn)                             |
|    - Train Sequential Model on scaled (X_train, y_train)          |
|    - Convert Keras model to float32 .tflite via TFLiteConverter   |
|    - Export model bytes + scaler_X / scaler_y to modelX_data.h    |
+-------------------------------------------------------------------+
                                 |
                                 v
+-------------------------------------------------------------------+
| 2. ESP32 Firmware (Arduino IDE / TFLite Micro C++)                |
|    - Load model_data[] into MicroInterpreter (32 KB Tensor Arena) |
|    - Listen for 5 comma-separated raw float inputs over Serial    |
|    - Scale inputs: x_scaled[i] = (x[i] - mean_X[i]) / scale_X[i]  |
|    - Execute interpreter->Invoke() on ESP32                       |
|    - Unscale outputs: y[i] = (y_scaled[i] * scale_y[i]) + mean_y  |
|    - Return 7 comma-separated float predictions over USB Serial   |
+-------------------------------------------------------------------+
                                 |
                                 v
+-------------------------------------------------------------------+
| 3. Hardware-in-the-Loop Benchmarking (Python + PySerial)          |
|    - Stream X_test samples to ESP32 over COM port (115200 baud)   |
|    - Collect y_pred_esp32 and compare against y_pred (Python)     |
|    - Compute MSE, MAE, Relative MSE, and R2 Score                 |
|    - Generate merged Python vs. ESP32 performance plots           |
+-------------------------------------------------------------------+
```

### Step 4.1: Exporting Model & Scalers to a C Header (`modelX_data.h`)
Because `scikit-learn` does not run on microcontrollers, feeding raw inputs into the TFLite model on the ESP32 would produce invalid outputs. A custom Python exporter embeds both the `.tflite` flatbuffer byte array (`alignas(8) const unsigned char model_data[]`) and the `StandardScaler` vectors directly into C float arrays:
* `scaler_X_mean[5]` and `scaler_X_scale[5]`
* `scaler_y_mean[7]` and `scaler_y_scale[7]`
* Reference test sample arrays (`test_input_sample[5]`, `expected_output_sample[7]`, `python_predicted_sample[7]`) for immediate single-sample verification at boot.

### Step 4.2: On-Device Scaling & Inference Equations
Inside the ESP32 C++ firmware, every incoming 5-element vector is normalized before `interpreter->Invoke()`:
$$X_{\text{scaled}}[i] = \frac{X_{\text{raw}}[i] - \mu_{X}[i]}{\sigma_{X}[i]} \quad \text{for } i \in \{0, \dots, 4\}$$

After inference completes, the 7 output tensor values are inverse-scaled before transmission:
$$\hat{y}[j] = \left(\hat{y}_{\text{scaled}}[j] \times \sigma_{y}[j]\right) + \mu_{y}[j] \quad \text{for } j \in \{0, \dots, 6\}$$

---

## 5. Resolving `TensorFlowLite_ESP32` Compiler Errors on ESP32 Arduino Core 3.x

When compiling the legacy `TensorFlowLite_ESP32` library under modern ESP32 Arduino Core 3.x (ESP-IDF v5.x / GCC 12+), three breaking compilation issues occur. Apply these three fixes once in your `Documents\Arduino\libraries\TensorFlowLite_ESP32` directory:

### Fix 1: `stl_emulation.h` Read-Only Member Assignment Error
* **Error:** `error: assignment of read-only member 'flatbuffers::span<T, Extent>::count_'`
* **Cause:** Older FlatBuffers declared `count_` as `const size_type count_;` inside `class span`, which modern C++ compilers reject inside `operator=`.
* **Solution:** Open `src\third_party\flatbuffers\stl_emulation.h`, locate `const size_type count_;` (around line 420–465), and remove `const` so it reads:
  ```cpp
  size_type count_;
  ```

### Fix 2: Legacy LCD Display & I2S Bus Driver Errors
* **Error:** `implicit declaration of function 'gpio_pad_select_gpio'`, `'periph_module_enable'`, `'LCD_WRITE_CMD'` inside `src\screen` and `src\bus`.
* **Cause:** The library bundles unused ESP-IDF v3 display drivers (`ssd1306`, `ili9341`, `st7789`, etc.) that Arduino's build system attempts to compile automatically.
* **Solution:** Delete the unused **`screen`** and **`bus`** folders from `Documents\Arduino\libraries\TensorFlowLite_ESP32\src\`.

### Fix 3: Private `operator delete` Error in Placement New
* **Error:** `'static void tflite::MicroErrorReporter::operator delete(void*)' is private within this context`
* **Cause:** The macro `TF_LITE_REMOVE_VIRTUAL_DELETE` was placed inside the `private:` section of classes instantiated via placement `new`.
* **Solution:** Open `src\tensorflow\lite\micro\compatibility.h` (lines 25–28) and add `public:` before `void operator delete(void* p) {}`:
  ```cpp
  #ifdef TF_LITE_STATIC_MEMORY
  #define TF_LITE_REMOVE_VIRTUAL_DELETE \
    public: void operator delete(void* p) {}
  #else
  ```

---

## 6. Evaluation Metrics & Merged Visualizations

### Metrics Computed (Python PC vs. ESP32 Hardware)
1. **Mean Squared Error (MSE):**
   $$\text{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2$$
2. **Mean Absolute Error (MAE):**
   $$\text{MAE} = \frac{1}{N} \sum_{i=1}^{N} \vert{}y_i - \hat{y}_i\vert{}$$
3. **Relative MSE:**
   $$\text{Relative MSE} = \frac{\text{MSE}}{\text{Var}(y_{\text{actual}})}$$
4. **Coefficient of Determination ($R^2$ Score — Model 3):**
   $$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$

### Generated Merged Plots (Per Model)
1. **Graph 1 — Merged Loss (MSE) Plot:** Plots the Python training loss (`yellow`) and validation loss (`blue`) across epochs, overlaid with horizontal benchmark lines comparing the scaled **Python Test Loss** and **ESP32 Hardware Test Loss**.
2. **Graph 2 — Merged Mean Absolute Error (MAE) Plot:** Plots the Python training MAE (`green`) and validation MAE (`red`) across epochs, overlaid with horizontal benchmark lines comparing the scaled **Python Test MAE** and **ESP32 Hardware Test MAE**.
3. **Graph 3 — Merged MSE vs. Relative MSE Bar Chart:** Compares unscaled `MSE` and `Relative MSE` side-by-side between Python (PC) and the ESP32 microcontroller (`skyblue` and `lightgreen`).

---

```text
project_root/
├── dataCombined1.csv               # Dataset (5 inputs, 7 outputs)
├── model1_train_and_test.py        # Model 1 (ReLU) training, export, and HIL plotting
├── model2_train_and_test.py        # Model 2 (ReLU) training, export, and HIL plotting
├── model3_train_and_test.py        # Model 3 (Sigmoid/Tanh + R2) training, export, and HIL plotting
├── esp32_model1_test/
│   ├── esp32_model1_test.ino       # ESP32 Serial inference firmware for Model 1
│   └── model_data.h                # Generated C array + StandardScaler for Model 1
├── esp32_model2_test/
│   ├── esp32_model2_test.ino       # ESP32 Serial inference firmware for Model 2
│   └── model2_data.h               # Generated C array + StandardScaler for Model 2
├── esp32_model3_test/
│   ├── esp32_model3_test.ino       # ESP32 Serial inference firmware for Model 3
│   └── model3_data.h               # Generated C array + StandardScaler for Model 3
└── README.md                       # Project documentation
```

---

## 7. How to Run the Project

1. **Install Python Dependencies:**
   ```bash
   pip install tensorflow pandas numpy scikit-learn matplotlib pyserial openpyxl
   ```
2. **Train & Export the Model Header:**
   Run the training and export script in Python to generate `model_data.h` (or `model2_data.h` / `model3_data.h`).
3. **Flash the ESP32 Firmware:**
   * Copy the generated `.h` file into the corresponding Arduino sketch folder.
   * Select **ESP32 Dev Module** and your COM port in the Arduino IDE, then click **Upload**.
   * **Close the Arduino Serial Monitor** after uploading so the COM port is free.
4. **Run Hardware-in-the-Loop Serial Evaluation & Plotting:**
   Update `SERIAL_PORT = "COM3"` (or your active port) in the Python script and run the Serial evaluation block to stream `X_test` to the ESP32, print the side-by-side metric table, and display the merged plots.