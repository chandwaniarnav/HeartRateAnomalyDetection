# HeartRateAnomalyDetection

This project demonstrates the deployment of a machine learning model for heart rate anomaly detection on an ESP32 microcontroller using the Wokwi simulator and Edge Impulse. The model is trained to classify heart rate anomalies, and the results are displayed on an LCD screen using the Adafruit ILI9341 display library.

## Table of Contents
- [Project Description](#project-description)
- [Prerequisites](#prerequisites)
- [Hardware Setup](#hardware-setup)
- [Software Setup](#software-setup)
- [Training the Model](#training-the-model)
- [Deploying the Model](#deploying-the-model)
- [Running the Simulator](#running-the-simulator)
- [Files Included](#files-included)
- [How to Run](#how-to-run)
- [License](#license)

## Project Description

This project uses a machine learning model created with **Edge Impulse** for heart rate anomaly detection. The trained model is then deployed on an **ESP32** microcontroller, which is simulated using the **Wokwi ESP32 Simulator**. 

### How It Works:
1. The trained model takes heart rate data as input and classifies it into different categories (normal or abnormal).
2. The ESP32 simulates the sensor data (in place of real heart rate sensor data).
3. The result (predicted class) is displayed on an LCD screen using the **Adafruit ILI9341** display.

### Features:
- Edge Impulse for model training and deployment.
- ESP32 simulation using Wokwi.
- TensorFlow Lite for microcontrollers to run the trained model.
- LCD screen to display predicted heart rate anomaly class.

## Prerequisites

Before running the project, you need to have the following:
- **VS Code** with **PlatformIO** extension installed.
- **Wokwi Simulator** for simulating the ESP32.
- **Edge Impulse Account** for training and exporting the model.
- **Arduino IDE** setup for ESP32 development.

### Libraries Used:
- `TensorFlowLite_ESP32`
- `Adafruit GFX`
- `Adafruit ILI9341`

## Hardware Setup

The hardware used for this project is an **ESP32** and an **ILI9341 TFT LCD** screen. The ESP32 simulates heart rate data in this example.

1. **ESP32**: 
   - Platform: ESP32.
   - Board: ESP32 Dev Board.
2. **TFT LCD**:
   - Connection: 3.3V, GND, SDA, SCL, CS, RESET, and DC pins connected to ESP32.

The LCD display shows the predicted class (normal or abnormal) based on the heart rate anomaly.

## Software Setup

1. Install **PlatformIO** extension in **VS Code**.
2. Clone this repository to your local machine or create a new PlatformIO project and copy the code into the project files.
3. Install the necessary libraries:
   - `TensorFlowLite_ESP32`
   - `Adafruit GFX`
   - `Adafruit ILI9341`

4. Modify the `platformio.ini` file to include the necessary configurations for your ESP32 board.

## Training the Model

The model is trained using **Edge Impulse**. Follow these steps to train the model:

1. Sign in to your **Edge Impulse** account.
2. Create a new project for **Heart Rate Anomaly Detection**.
3. Collect heart rate data using the Edge Impulse Data Acquisition tools or use any existing dataset for heart rate anomaly detection.
4. Train the model and test it to ensure its accuracy.
5. After training the model, export the model as a TensorFlow Lite model and obtain the `.tflite` file.
6. Download the trained model and integrate it into the ESP32 project.

## Deploying the Model

1. After obtaining the TensorFlow Lite model (`tflite_learn_2`), include it in your project directory.
2. Update the `main.cpp` to include the generated model (`tflite_learn_2.h`).
3. The model file is included in the project as a header (`tflite_learn_2.h`), and the TensorFlow Lite Micro library is used to deploy it on the ESP32.

## Running the Simulator

1. Open the project in **VS Code** with **PlatformIO**.
2. Make sure the board configuration in `platformio.ini` matches your ESP32 board.
3. Click **Build** in PlatformIO to compile the code.
4. Use the **Wokwi Simulator** to simulate the ESP32 and LCD screen:
   - Upload the firmware to the simulator.
   - The simulated ESP32 will display the heart rate anomaly classification on the TFT LCD.

## Files Included

1. **main.cpp**: Contains the code for setting up the ESP32, loading the model, and making predictions.
2. **tflite_learn_2.h**: Contains the exported TensorFlow Lite model.
3. **platformio.ini**: PlatformIO configuration file with library dependencies and build settings.
4. **wokwi.toml**: Wokwi configuration file for ESP32 simulation.

## How to Run

1. Clone this repository to your local machine.
2. Open the project folder in **VS Code** and make sure PlatformIO is installed.
3. Build the project by clicking **Build** in PlatformIO.
4. Simulate the ESP32 using **Wokwi**:
   - Open the `wokwi.toml` configuration file.
   - Upload the generated firmware to the Wokwi ESP32 simulator.
5. Observe the output on the simulated TFT screen for the predicted class.

## License

This project is open-source and available under the [MIT License](LICENSE).

---

### Acknowledgements

- **Edge Impulse**: For providing the tools to train and deploy the machine learning model.
- **TensorFlow Lite**: For providing the microcontroller-optimized version of the machine learning framework.
- **Wokwi**: For providing the ESP32 simulator.

For further information on **Edge Impulse**, visit: [Edge Impulse](https://www.edgeimpulse.com/)
