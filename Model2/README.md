# Model 2: TinyML Regression on ESP32

Model 2 is a multi-output regression model that predicts seven target values
from five measured/input features. It is trained with Keras on a computer and
deployed to an ESP32 as a float32 TensorFlow Lite Micro model. After deployment,
the ESP32 performs preprocessing, inference, and output postprocessing locally;
the host only needs to send input values over the serial connection.

The complete workflow is:

1. Load and clean the training dataset.
2. Train and evaluate the neural network in Python.
3. Export the trained Keras model to TensorFlow Lite format.
4. Generate a C/C++ header containing the model and scaler parameters.
5. Compile the header into the Arduino ESP32 sketch.
6. Stream test inputs to the board and compare ESP32 predictions with Python
   predictions.

## Model description

### Input features

Model 2 expects five values in this exact order:

1. `frequency`
2. `Rimp`
3. `Ximp`
4. `Directivity`
5. `S11_dB`

The seven regression targets are selected by `train2.py` using the first seven
columns of the dataset. Their names and meaning are therefore defined by the
dataset header and remain in the same column order during training and
inference.

### Neural-network architecture

| Layer | Configuration |
|---|---|
| Input | 5 features |
| Dense | 64 units, ReLU activation |
| Dense | 128 units, ReLU activation |
| Dense | 64 units, ReLU activation |
| Output | 7 units, linear activation |

The model is compiled with:

- **Optimizer:** Adam
- **Loss:** Mean squared error (MSE)
- **Metric:** Mean absolute error (MAE)

Training uses an 80/20 train/test split, a 20% validation split from the
training data, a maximum of 100 epochs, a batch size of 64, and early stopping
with patience of 10 epochs. The best validation weights are restored.

### Scaling

`StandardScaler` is fitted separately to the five inputs and seven targets.
The generated `model2_data.h` stores the resulting means and scales. The
ESP32 applies exactly the same transformations as the Python code:

```text
scaled_input[i] = (raw_input[i] - scaler_X_mean[i]) / scaler_X_scale[i]
raw_output[i] = scaled_output[i] * scaler_y_scale[i] + scaler_y_mean[i]
```

The exported model is float32 and is not integer-quantized. This preserves the
same floating-point inference representation used by the Keras model.

## Project files

```text
Model2/
├── dataCombined1.csv              Training dataset
├── train2.py                      Training, export, and ESP32 benchmark script
├── model2_data.h                  Generated model header
└── sketch_oct31a/
    ├── sketch_oct31a.ino          ESP32 TensorFlow Lite Micro application
    └── model2_data.h              Header included by the Arduino sketch
```

`train2.py` writes the generated header to the Model2 directory. The Arduino
sketch includes the copy located in `sketch_oct31a`, so copy the regenerated
header there whenever the model or dataset changes.

## Train and export Model 2

From the `Model2` directory, create a Python environment and install the
required packages:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install pandas numpy matplotlib scikit-learn tensorflow pyserial openpyxl
```

Run the training and export script:

```bash
python train2.py
```

The script:

- loads the dataset and removes rows containing missing values;
- selects the five input features and seven target columns;
- trains the dense regression network;
- reports Python MSE, MAE, and relative MSE;
- converts the trained Keras model to float32 TensorFlow Lite bytes;
- generates `model2_data.h` with the model bytes, scaler constants, and a
  reference test sample; and
- optionally streams up to 200 test samples to the ESP32 for hardware
  evaluation.

The serial benchmark uses `COM6` and 115200 baud by default. Change
`SERIAL_PORT` in `train2.py` to the COM port assigned to your ESP32. Set
`MAX_TEST_SAMPLES = len(X_test)` to benchmark the complete test set.

After the script completes, copy the generated header:

```text
Model2\model2_data.h
    ↓
Model2\sketch_oct31a\model2_data.h
```

Do not manually edit the generated model header. Regenerate and recopy it after
every model or dataset update.

> **Dataset note:** the file is named `.csv`, but `train2.py` currently loads it
> with `pandas.read_excel`. Keep the existing spreadsheet-formatted file or
> change the loader to `pd.read_csv` after converting it to a real CSV file.

## Deploy Model 2 with TinyML on ESP32

### Install the Arduino environment

1. Install the Arduino IDE.
2. Install the ESP32 board package through **Boards Manager**.
3. Select the correct ESP32 board and COM port.
4. Install the `TensorFlowLite_ESP32` library and its dependencies.

The sketch registers the two operators required by this network:

- `FullyConnected`
- `Relu`

It allocates a 32 KiB TensorFlow Lite Micro tensor arena:

```cpp
constexpr int kTensorArenaSize = 32 * 1024;
```

If tensor allocation fails after changing the model or library version, increase
the arena size if the selected ESP32 has enough available RAM.

### Upload the sketch

Open `sketch_oct31a\sketch_oct31a.ino` in Arduino IDE, confirm the board and
port, and upload it. On startup, the board sends:

```text
ESP32_MODEL2_READY
```

This confirms that the Model2 sketch has initialized its interpreter and is
ready to receive inference requests.

### Serial inference protocol

The sketch communicates at **115200 baud**. Send one newline-terminated line
containing five comma-separated raw feature values:

```text
frequency,Rimp,Ximp,Directivity,S11_dB
```

Example:

```text
915.0,12.4,-3.1,0.82,-18.6
```

For each valid line, the ESP32:

1. parses the five raw values;
2. standardizes them using `scaler_X_mean` and `scaler_X_scale`;
3. invokes the embedded TensorFlow Lite Micro model;
4. inverse-standardizes the seven model outputs; and
5. prints one comma-separated line containing seven predictions.

If model invocation fails, the sketch prints:

```text
ERR
```

Malformed input lines are ignored. The startup message is not a prediction and
must be skipped by host-side code before parsing prediction rows.

## Validate ESP32 inference

With the Model2 sketch uploaded and the correct COM port configured, run:

```bash
python train2.py
```

The script sends the selected Python test samples to the ESP32, reads seven
outputs per sample, and reports:

- ESP32 MSE
- ESP32 MAE
- ESP32 relative MSE
- Python-versus-ESP32 metrics on the same test subset

Small differences are expected because Python and TensorFlow Lite Micro execute
on different runtimes and hardware. Large differences commonly result from
using an outdated `model2_data.h`, changing the input order, sending scaled
values instead of raw values, or using scaler constants from another model.

## Troubleshooting

- **Header not found during compilation:** run `train2.py` and copy
  `model2_data.h` into `sketch_oct31a`.
- **No serial response:** check the COM port, use 115200 baud, send a newline,
  and wait for `ESP32_MODEL2_READY`.
- **Incorrect predictions:** verify the five input values are in the documented
  order and are unscaled raw measurements.
- **Tensor allocation failure:** increase `kTensorArenaSize` when memory allows.
- **Dataset loading error:** reconcile the `.csv` extension with the current
  `pd.read_excel` loader, or convert the data and update the loader.
- **Benchmark parsing warnings:** ensure the host ignores the startup marker and
  reads exactly seven comma-separated numeric values for each request.

## Limitations

- Updating the model requires regenerating the header and recompiling the
  ESP32 sketch.
- The current serial protocol is intended for development and benchmarking, not
  a production packet-framed communication protocol.
- Target names are not separately encoded in the Arduino sketch; the output
  order is the first seven dataset columns used by `train2.py`.