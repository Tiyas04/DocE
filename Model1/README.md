# Model 1: TinyML Regression on ESP32

Model 1 is a small, fully connected neural network for multi-output regression. It
maps five measured/input features to seven target values and runs the trained
model locally on an ESP32 using TensorFlow Lite for Microcontrollers.

The project is designed for an edge-inference workflow:

1. Train and evaluate the model on a computer with Python.
2. Export the Keras model to a float32 TensorFlow Lite model.
3. Generate a C/C++ header containing the model bytes and preprocessing values.
4. Compile that header into an Arduino ESP32 sketch.
5. Send feature values over the serial port and receive seven predictions from
   the ESP32.

## Model description

### Inputs

The model expects five input features in this order:

1. `frequency`
2. `Rimp`
3. `Ximp`
4. `Directivity`
5. `S11_dB`

The training script reads the first seven columns of the dataset as the seven
regression targets. Their column names are therefore determined by the dataset
header in `dataCombined1.csv` (the file currently contains spreadsheet-formatted
data despite its `.csv` extension).

### Network architecture

The model is a dense feed-forward network:

| Layer | Configuration |
|---|---|
| Input | 5 features |
| Dense | 64 units, ReLU |
| Dense | 128 units, ReLU |
| Dense | 64 units, ReLU |
| Output | 7 linear values |

The model uses the Adam optimizer, mean squared error (MSE) loss, and mean
absolute error (MAE) as a metric. Training uses an 80/20 train/test split,
20% of the training portion for validation, a maximum of 100 epochs, a batch
size of 64, and early stopping with a patience of 10 epochs.

Both inputs and targets are standardized with scikit-learn `StandardScaler`.
The generated `model_data.h` stores the scaler means and scales so the ESP32
performs the same preprocessing and postprocessing as Python:

```text
scaled_input[i] = (input[i] - scaler_X_mean[i]) / scaler_X_scale[i]
output[i] = scaled_output[i] * scaler_y_scale[i] + scaler_y_mean[i]
```

The exported model is float32 rather than quantized. This keeps the ESP32
inference path consistent with the Python Keras model and requires a float-capable
TensorFlow Lite Micro runtime.

## Project files

```text
Model1/
├── dataCombined1.csv              Training dataset
├── train1.py                      Training, export, and serial benchmark script
├── model_data.h                   Generated header (created by train1.py)
└── sketch_oct31a/
    ├── sketch_oct31a.ino          ESP32 TensorFlow Lite Micro application
    └── model_data.h               Header compiled by the Arduino sketch
```

`train1.py` generates `model_data.h` beside the script. The Arduino sketch
includes the copy in `sketch_oct31a`, so copy the newly generated header into
that folder before compiling if the model or dataset changes.

## Train and export the model

Create a Python environment and install the dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install pandas numpy matplotlib scikit-learn tensorflow pyserial openpyxl
```

From the `Model1` directory, run:

```bash
python train1.py
```

The script:

- loads and removes rows containing missing values;
- trains the Keras regression model;
- reports Python MSE, MAE, and relative MSE;
- converts the trained model to float32 `.tflite` bytes;
- writes the scaler parameters, reference samples, and model bytes to
  `model_data.h`; and
- optionally streams test samples to the ESP32 for a hardware comparison.

The serial benchmark section uses `COM6` by default. Change `SERIAL_PORT` in
`train1.py` to the port assigned to the board, or temporarily disable/comment
out that section when only a model export is needed.

After training, copy the generated header:

```text
Model1\model_data.h
    ↓
Model1\sketch_oct31a\model_data.h
```

Do not edit the generated header manually; rerun the training script whenever
the model or dataset changes.

## Deploy with TinyML on ESP32

### 1. Install the Arduino toolchain

1. Install the Arduino IDE.
2. Add the ESP32 board package through **Boards Manager**.
3. Select the appropriate ESP32 board and COM port.
4. Install the `TensorFlowLite_ESP32` library, including its dependencies, from
   the Arduino Library Manager or the library source used by the project.

The sketch uses TensorFlow Lite Micro operators for:

- fully connected (`FullyConnected`) layers; and
- ReLU activation (`Relu`).

The Tensor arena is configured as 32 KiB:

```cpp
constexpr int kTensorArenaSize = 32 * 1024;
```

Increase this value if tensor allocation fails on a different model or library
build.

### 2. Open and upload the sketch

Open `sketch_oct31a\sketch_oct31a.ino` in Arduino IDE, confirm the board and
port, and click **Upload**. After reset, the ESP32 prints:

```text
ESP32_READY
```

### 3. Run inference over serial

The sketch communicates at **115200 baud**. Send one newline-terminated line
containing five comma-separated raw feature values:

```text
frequency,Rimp,Ximp,Directivity,S11_dB
```

For example:

```text
915.0,12.4,-3.1,0.82,-18.6
```

For every valid input line, the ESP32:

1. parses the five values;
2. standardizes them using `scaler_X_mean` and `scaler_X_scale`;
3. invokes the TensorFlow Lite Micro model;
4. inverse-standardizes the seven outputs; and
5. prints one comma-separated line containing seven predictions.

An invocation failure produces:

```text
ERR
```

Malformed input lines are ignored by the current sketch. The first startup
message (`ESP32_READY`) is not a prediction and should be discarded by host
benchmarking code before parsing output rows.

## Validate Python and ESP32 results

With the sketch running and the correct `SERIAL_PORT` configured in
`train1.py`, run:

```bash
python train1.py
```

The serial benchmark sends up to 200 rows from the Python test set by default
(`MAX_TEST_SAMPLES = 200`), receives seven predictions for each row, and reports
Python-versus-ESP32 MSE, MAE, and relative MSE. Set
`MAX_TEST_SAMPLES = len(X_test)` to benchmark the complete test set.

Small numerical differences between Python and ESP32 are expected because the
model is executed in a different runtime and hardware environment. Large
differences usually indicate that the wrong `model_data.h` was copied, the
input feature order is incorrect, or the scaler constants do not match the
trained model.

## Troubleshooting

- **`dataCombined1.csv` cannot be read as CSV:** the training script currently
  calls `pandas.read_excel`; keep the spreadsheet-formatted file or change the
  loader to `read_csv` after converting the dataset to a real CSV file.
- **`model_data.h` is missing during compilation:** run `train1.py` and copy the
  generated header into `sketch_oct31a`.
- **Tensor allocation fails:** increase `kTensorArenaSize` if the board has
  sufficient RAM.
- **No serial response:** check the selected COM port, use 115200 baud, include
  a newline after every five-value input, and wait for `ESP32_READY`.
- **Predictions are implausible:** verify the five-value order and ensure raw
  values, not already-standardized values, are sent to the ESP32.

## Limitations

- The model and scaler are embedded in program memory through a generated C
  array, so recompilation is required for model updates.
- The current serial protocol is intended for testing and host-to-board
  communication; it is not a production transport or packet-framed protocol.
- The target names are not declared separately in the sketch. They correspond
  to the first seven dataset columns in their existing order.