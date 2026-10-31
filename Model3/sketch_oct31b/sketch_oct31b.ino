#include <Arduino.h>
#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model3_data.h" // Model 3 weights and scalers

constexpr int kTensorArenaSize = 32 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input_tensor = nullptr;
TfLiteTensor* output_tensor = nullptr;

constexpr int NUM_INPUTS = 5;
constexpr int NUM_OUTPUTS = 7;

void setup() {
  Serial.begin(115200);

  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  model = tflite::GetModel(model_data);
  
  // Register Dense, Sigmoid (Logistic), and Tanh operations for Model 3
  static tflite::MicroMutableOpResolver<3> resolver;
  resolver.AddFullyConnected();
  resolver.AddLogistic();
  resolver.AddTanh();

  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;
  interpreter->AllocateTensors();

  input_tensor = interpreter->input(0);
  output_tensor = interpreter->output(0);

  Serial.println("ESP32_MODEL3_READY");
}

void loop() {
  // Listen for 5 comma-separated floats from Python: "f0,f1,f2,f3,f4\n"
  if (Serial.available() > 0) {
    String line = Serial.readStringUntil('\n');
    line.trim();
    if (line.length() == 0) return;

    float raw_inputs[NUM_INPUTS];
    int start_idx = 0;
    for (int i = 0; i < NUM_INPUTS; i++) {
      int comma_idx = line.indexOf(',', start_idx);
      if (comma_idx == -1 && i < NUM_INPUTS - 1) return;
      String token = (comma_idx == -1) ? line.substring(start_idx)
                                       : line.substring(start_idx, comma_idx);
      raw_inputs[i] = token.toFloat();
      start_idx = comma_idx + 1;
    }

    // 1. Normalize the 5 inputs using Model 3 scaler_X
    for (int i = 0; i < NUM_INPUTS; i++) {
      input_tensor->data.f[i] = (raw_inputs[i] - scaler_X_mean[i]) / scaler_X_scale[i];
    }

    // 2. Run inference on ESP32
    if (interpreter->Invoke() != kTfLiteOk) {
      Serial.println("ERR");
      return;
    }

    // 3. Inverse-scale the 7 outputs using Model 3 scaler_y and send over Serial
    for (int i = 0; i < NUM_OUTPUTS; i++) {
      float y_scaled = output_tensor->data.f[i];
      float y_val = (y_scaled * scaler_y_scale[i]) + scaler_y_mean[i];
      Serial.print(y_val, 6);
      if (i < NUM_OUTPUTS - 1) Serial.print(",");
    }
    Serial.println();
  }
}