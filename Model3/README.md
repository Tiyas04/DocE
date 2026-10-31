# Model 3: TinyML Regression on ESP32

Model 3 is a multi-output regression neural network that maps five input
features to seven target values. It is trained with Keras and scikit-learn,
converted to a float32 TensorFlow Lite model, and deployed to an ESP32 with
TensorFlow Lite for Microcontrollers.

The ESP32 performs the complete inference pipeline at the edge:

1. Receive five raw feature values over serial.
2. Standardize the inputs using the training scaler.
3. Run the embedded neural network.
4. Inverse-standardize the seven outputs.
5. Return the predictions as a comma-separated serial line.

## Model description

### Input features

The model expects five input features in the following order:

1. `frequency`
2. `Rimp`
3. `Ximp`
4. `Directivity`
5. `S11_dB`

The seven regression targets are selected from the first seven columns of the
dataset by `train3.py`. Their names and meanings are defined by the dataset
header, and their output order is preserved during training and deployment.

### Network architecture

| Layer | Configuration |
|---|---|
| Input | 5 features |
| Dense | 64 units, sigmoid activation |
| Dense | 128 units, tanh activation |
| Dense | 64 units, tanh activation |
| Output | 7 units, linear activation |

The model uses:

- **Optimizer:** Adam
- **Loss:** Mean squared error (MSE)
- **Metric:** Mean absolute error (MAE)

The training script uses an 80/20 train/test split, a 20% validation split
from the training portion, a maximum of 100 epochs, a batch size of 64, and
early stopping with patience of 10 epochs. The best validation weights are
restored.

In addition to MSE, MAE, and relative MSE, the script reports R-squared (R2)
for the Python training and test predictions. The ESP32 benchmark also reports
test-set R2.

### Input and output scaling

The five inputs and seven targets are standardized independently with
scikit-learn `StandardScaler`. The generated `model3_data.h` stores the
scaler means and scales used by both Python and the ESP32.

```text
scaled_input[i] = (raw_input[i] - scaler_X_mean[i]) / scaler_X_scale[i]
raw_output[i] = scaled_output[i] * scaler_y_scale[i] + scaler_y_mean[i]
```

The exported TensorFlow Lite model uses float32 tensors. It is not integer
quantized, so the ESP32 sketch uses floating-point input and output tensors.

## Project files

```text
Model3/
├── dataCombined1.csv              Training dataset
├── train3.py                      Training, export, and ESP32 benchmark script
├── model3_data.h                  Generated model/scaler header
└── sketch_oct31b/
    ├── sketch_oct31b.ino          ESP32 TensorFlow Lite Micro application
    └── model3_data.h              Header included by the Arduino sketch
```

`train3.py` creates `model3_data.h` beside the training script. The Arduino
sketch uses the copy in `sketch_oct31b`, so regenerate and copy the header
whenever the model or training data changes:

```text
Model3\model3_data.h
    ↓
Model3\sketch_oct31b\model3_data.h
```

The generated header contains:

- input scaler means and scales;
- output scaler means and scales;
- one reference test input;
- the expected output for that reference input;
- the Python prediction for that reference input; and
- the TensorFlow Lite model byte array.

## Train and export Model 3

From the `Model3` directory, create a Python environment and install the
dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install pandas numpy matplotlib scikit-learn tensorflow pyserial openpyxl
```

Run the training and export script:

```bash
python train3.py
```

The script loads the dataset, drops rows containing missing values, trains the
network, evaluates the Python model, exports the model to float32 TensorFlow
Lite, and generates `model3_data.h`.

The script also contains an ESP32 hardware benchmark. It sends up to 200 test
samples by default:

```python
MAX_TEST_SAMPLES = 200
```

Set this to `len(X_test)` to stream the complete test set. The default serial
configuration is:

```python
SERIAL_PORT = "COM6"
BAUD_RATE = 115200
```

Change `SERIAL_PORT` to the COM port assigned to the ESP32 before running the
benchmark. If only model export is required, do not run the serial benchmark
section until the board is connected and programmed.

> **Dataset note:** the input file is named `dataCombined1.csv`, but the
> training script currently loads it with `pandas.read_excel`. Keep the existing
> spreadsheet-formatted file or change the loader to `pd.read_csv` after
> converting the file to a true CSV.

## Deploy Model 3 with TinyML on ESP32

### 1. Prepare Arduino and TensorFlow Lite Micro

1. Install the Arduino IDE.
2. Install the ESP32 board package through **Boards Manager**.
3. Select the correct ESP32 board and COM port.
4. Install the `TensorFlowLite_ESP32` library and its dependencies.
5. Copy the generated `model3_data.h` into the `sketch_oct31b` directory.

Model 3 requires these TensorFlow Lite Micro operators:

- `FullyConnected` for the dense layers;
- `Logistic` for the sigmoid activation; and
- `Tanh` for the tanh activations.

The sketch registers them with a resolver sized for three operators:

```cpp
static tflite::MicroMutableOpResolver<3> resolver;
resolver.AddFullyConnected();
resolver.AddLogistic();
resolver.AddTanh();
```

The TensorFlow Lite Micro tensor arena is set to 32 KiB:

```cpp
constexpr int kTensorArenaSize = 32 * 1024;
```

If tensor allocation fails after changing the model or library version, increase
the arena size if the ESP32 has sufficient available RAM.

### 2. Upload the ESP32 sketch

Open `sketch_oct31b\sketch_oct31b.ino` in Arduino IDE, verify the board and
port, and upload the sketch. After startup, the board prints:

```text
ESP32_MODEL3_READY
```

This indicates that the Model3 TensorFlow Lite Micro interpreter has been
initialized.

### 3. Send inference requests

Use a serial connection at **115200 baud**. Send one newline-terminated line
containing five comma-separated raw values in the documented feature order:

```text
frequency,Rimp,Ximp,Directivity,S11_dB
```

Example:

```text
915.0,12.4,-3.1,0.82,-18.6
```

For each valid line, the ESP32:

1. parses the five values;
2. standardizes them using `scaler_X_mean` and `scaler_X_scale`;
3. invokes the embedded Model3 TensorFlow Lite model;
4. inverse-standardizes the seven output values; and
5. prints seven comma-separated predictions with six decimal places.

If inference fails, the sketch prints:

```text
ERR
```

Malformed input lines are ignored. `ESP32_MODEL3_READY` is a startup status
message, not a prediction, and must be skipped by host-side parsers.

## Validate the ESP32 deployment

After uploading the sketch and setting the correct COM port, run:

```bash
python train3.py
```

The benchmark sends the selected Python test samples to the ESP32 and compares
the returned predictions with the known test targets. It reports:

- ESP32 MSE;
- ESP32 MAE;
- ESP32 relative MSE;
- ESP32 test-set R2; and
- Python-versus-ESP32 results for the same test subset.

Small numerical differences are expected because Python Keras and TensorFlow
Lite Micro run on different runtimes and hardware. Large differences commonly
indicate an outdated `model3_data.h`, incorrect input ordering, mismatched
scaler values, or already-standardized values being sent to the board.

## Troubleshooting

- **Compilation cannot find `model3_data.h`:** run `train3.py` and copy the
  generated header into `sketch_oct31b`.
- **Model invocation fails:** check that the sketch registers
  `FullyConnected`, `Logistic`, and `Tanh`, and increase the tensor arena if
  required.
- **No serial response:** check the COM port, use 115200 baud, send a newline,
  and wait for `ESP32_MODEL3_READY`.
- **Predictions are incorrect:** verify the five raw inputs are in the exact
  documented order and that the header was generated from the same model.
- **Benchmark parsing warnings:** ignore the startup status line and parse only
  lines containing seven comma-separated numeric values.
- **Dataset loading errors:** reconcile the `.csv` extension with the current
  `pd.read_excel` loader or convert the data and update the loader.

## Limitations

- Updating the model requires regenerating `model3_data.h` and recompiling the
  ESP32 sketch.
- The serial protocol is intended for development and benchmarking, not
  production-grade framed communication.
- Target names are not separately encoded in the Arduino sketch; the output
  order follows the first seven dataset columns used by `train3.py`.