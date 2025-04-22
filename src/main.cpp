#include <Arduino.h>
#include <TensorFlowLite_ESP32.h>
#include <Adafruit_GFX.h>
#include <Adafruit_ILI9341.h>
#include "../include/tflite-model/tflite_learn_2.h"
#include "tensorflow/lite/schema/schema_utils.h"
#include "tensorflow/lite/micro/micro_allocator.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_op_resolver.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/kernels/conv.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_utils.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/builtin_op_data.h"
#include "tensorflow/lite/portable_type_to_tflitetype.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/core/api/tensor_utils.h"
#include "tensorflow/lite/core/api/op_resolver.h"
#include "tensorflow/lite/core/api/flatbuffer_conversions.h"
#include "tensorflow/lite/schema/schema_utils.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/micro/test_helper_custom_ops.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "../include/edge-impulse-sdk/tensorflow/lite/micro/all_ops_resolver.h"
float DequantizeUint8(uint8_t value, float scale, int zero_point) {
    return (value - zero_point) * scale;
}
void ApplySoftmax(float* values, int size) {
    float sum = 0.0;
    for (int i = 0; i < size; i++) {
        values[i] = exp(values[i]);
        sum += values[i];
    }
    for (int i = 0; i < size; i++) {
        values[i] /= sum;
    }
}
#define TFT_CS     15
#define TFT_RST    4
#define TFT_DC     2
Adafruit_ILI9341 tft = Adafruit_ILI9341(TFT_CS, TFT_DC, TFT_RST);
const tflite::Model* model;
tflite::MicroInterpreter* interpreter;
TfLiteTensor* input;
TfLiteTensor* output;
tflite::MicroErrorReporter micro_error_reporter;
static tflite::MicroAllocator* allocator;
constexpr int kTensorArenaSize = 1024 * 10; 
uint8_t tensor_arena[kTensorArenaSize];
void setup() {
  Serial.begin(115200);  
  tft.begin();
  tft.setRotation(3);  
  tft.fillScreen(ILI9341_BLACK);
  model = tflite::GetModel(tflite_learn_2);
  tflite::AllOpsResolver resolver;
  allocator = tflite::MicroAllocator::Create(tensor_arena, kTensorArenaSize, &micro_error_reporter);
  static tflite::MicroInterpreter static_interpreter(model, resolver, allocator, &micro_error_reporter);
  interpreter = &static_interpreter;
  interpreter->AllocateTensors();
  input = interpreter->input(0); 
  output = interpreter->output(0);  
}
int classify(float *input_data) {
  for (int i = 0; i < input->dims->data[1]; i++) {
    input->data.f[i] = input_data[i];
    Serial.print(input->data.f[i]);
  }
  interpreter->Invoke();
  float output_scale = output->params.scale;
  int output_zero_point = output->params.zero_point;

  float dequantized_values[output->dims->data[1]];
    for (int i = 0; i < output->dims->data[1]; i++) {
      uint8_t raw_value = output->data.uint8[i];
      dequantized_values[i] = DequantizeUint8(raw_value, output_scale, output_zero_point);
    }
    ApplySoftmax(dequantized_values, output->dims->data[1]);
    Serial.println("Output probabilities:");
    int predicted_class = -1;
    float max_score = -9999;
    for (int i = 0; i < output->dims->data[1]; i++) {
        Serial.print("Class ");
        Serial.print(i);
        Serial.print(": ");
        Serial.println(dequantized_values[i]);
        if (dequantized_values[i] > max_score) {
            max_score = dequantized_values[i];
            predicted_class = i;
        }
    }
    return predicted_class;
}
void loop() {
  // Simulate input (replace with actual sensor data in a real application)
  /*float input_data[187] = {
1,0.742331266,0.07668712,0.134969324,0.122699387,0.125766873,0.119631901,0.125766873,0.110429451,0.122699387,0.113496929,0.125766873,0.113496929,0.119631901,0.110429451,0.122699387,0.119631901,0.125766873,0.107361965,0.119631901,0.110429451,0.110429451,0.110429451,0.119631901,0.116564415,0.122699387,0.122699387,0.125766873,0.107361965,0.116564415,0.104294479,0.110429451,0.098159507,0.110429451,0.098159507,0.116564415,0.119631901,0.144171774,0.147239268,0.16257669,0.159509197,0.177914113,0.171779141,0.184049085,0.174846619,0.190184042,0.180981591,0.184049085,0.168711662,0.180981591,0.16257669,0.171779141,0.16257669,0.177914113,0.159509197,0.156441718,0.147239268,0.15337424,0.141104296,0.134969324,0.128834352,0.147239268,0.138036802,0.147239268,0.141104296,0.144171774,0.134969324,0.144171774,0.138036802,0.147239268,0.141104296,0.150306746,0.15337424,0.171779141,0.174846619,0.196319014,0.196319014,0.208588958,0.196319014,0.202453986,0.193251535,0.199386507,0.208588958,0.205521479,0.159509197,0.138036802,0.122699387,0.128834352,0.116564415,0.128834352,0.119631901,0.138036802,0.110429451,0.107361965,0.042944785,0,0.052147239,0.291411042,0.628834367,0.960122705,0.739263833,0.08588957,0.095092021,0.107361965,0.098159507,0.104294479,0.088957056,0.101226993,0.101226993,0.101226993,0.088957056,0.095092021,0.073619634,0.092024542,0.082822084,0.088957056,0.082822084,0.092024542,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
  };*/
  float input_data[187]={0.891525447,0.501694918,0.538983047,0.552542388,0.552542388,0.52203387,0.494915247,0.406779647,0.277966112,0.142372876,0,0.044067796,0.074576274,0.074576274,0.125423729,0.17288135,0.149152547,0.128813565,0.128813565,0.142372876,0.166101694,0.17288135,0.166101694,0.176271185,0.169491529,0.169491529,0.155932203,0.132203385,0.108474575,0.108474575,0.11186441,0.091525421,0.118644066,0.11186441,0.128813565,0.149152547,0.169491529,0.196610168,0.223728821,0.240677968,0.271186441,0.274576277,0.284745753,0.277966112,0.281355917,0.271186441,0.261016935,0.250847459,0.237288132,0.240677968,0.233898312,0.233898312,0.237288132,0.233898312,0.233898312,0.233898312,0.233898312,0.233898312,0.233898312,0.21694915,0.223728821,0.21694915,0.223728821,0.21694915,0.213559315,0.210169494,0.206779659,0.206779659,0.203389823,0.206779659,0.210169494,0.206779659,0.196610168,0.193220332,0.203389823,0.196610168,0.186440676,0.196610168,0.193220332,0.193220332,0.186440676,0.183050841,0.189830512,0.189830512,0.189830512,0.186440676,0.189830512,0.186440676,0.189830512,0.193220332,0.193220332,0.193220332,0.200000003,0.203389823,0.213559315,0.21694915,0.250847459,0.396610171,1,0.515254259,0.518644094,0.508474588,0.528813541,0.498305082,0.501694918,0.447457641,0.349152535,0.230508476,0.057627119,0.016949153,0.091525421,0.091525421,0.088135593,0.115254238,0.166101694,0.135593221,0.138983056,0.135593221,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
  };
  int predicted_class = classify(input_data);
  tft.fillScreen(ILI9341_BLACK);
  tft.setTextColor(ILI9341_WHITE);
  tft.setTextSize(2);
  tft.setCursor(10, 10);
  tft.print("Predicted class: ");
  tft.println(predicted_class);
  delay(2000);
}