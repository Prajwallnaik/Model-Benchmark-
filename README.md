# Tomato Disease Detection & Model Benchmark

This repository contains a comprehensive deep learning framework designed to detect Tomato Early Blight and Tomato Late Blight diseases. It features a benchmarking suite to evaluate the performance of state-of-the-art convolutional neural network architectures—specifically MobileNetV2, ResNet50, and EfficientNetB0—and includes a production-ready FastAPI web application for real-time inference.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95%2B-009688)
![License](https://img.shields.io/badge/License-MIT-green)

## Technical Stack

The project utilizes the following technologies and libraries:

*   **Deep Learning Framework:** PyTorch, Torchvision
*   **Model Architectures:** MobileNetV2, ResNet50, EfficientNetB0 (Pre-trained on ImageNet)
*   **Backend API:** FastAPI, Uvicorn
*   **Data Processing:** Pandora, NumPy, Scikit-Learn (Metrics)
*   **Image Processing:** Pillow (PIL)
*   **Frontend:** HTML5, CSS3, Jinja2 Templating
*   **Deployment Environment:** Python 3.9+

## Project Structure

```
model_benchmark/
├── artifacts/          # Serialized model weights (.pth files)
├── data/               # Dataset directory (Train/Validation splits)
├── results/            # Benchmark performance metrics (JSON)
├── templates/          # HTML templates for the web interface
├── app.py              # Flask Backend (Legacy implementation)
├── benchmark.py        # Core training and benchmarking script
├── main.py             # FastAPI Backend entry point
├── requirements.txt    # Python dependencies
└── split_data.py       # Data preparation utility
```

## Key Features

*   **Comprehensive Benchmarking:** Automated training and evaluation of multiple architectures to compare accuracy, inference latency, FLOPs, and model size.
*   **Real-time Inference:** A lightweight web interface allowing users to upload leaf images and receive immediate disease predictions.
*   **Confidence Thresholding:** Implements logic to flag low-confidence predictions (below 70%) as unknown to minimize false positives on out-of-distribution data.
*   **Artifact Management:** Automatically versions and saves the best-performing model weights and corresponding training metrics.

## Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/Prajwallnaik/Model-Benchmark-.git
    cd Model-Benchmark-
    ```

2.  **Environment Setup**
    It is recommended to use a virtual environment.
    ```bash
    python -m venv venv
    # Activate on Windows:
    venv\Scripts\activate
    # Activate on macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Running the Benchmark

Execute the benchmarking script to train the models and generate performance reports.

```bash
python benchmark.py
```

Outputs will be stored in:
*   `artifacts/`: Trained model files (e.g., `MobileNet.pth`)
*   `results/`: Metrics JSON files (e.g., `metrics_mobilenet.json`)

### 2. Launching the Web Application

Start the FastAPI server to access the inference interface.

```bash
uvicorn main:app --reload
```

Open a web browser and navigate to: [http://127.0.0.1:8000](http://127.0.0.1:8000)

## Performance Results

| Model Architecture | Accuracy | Inference Time (CPU) | Model Size |
| :--- | :--- | :--- | :--- |
| **MobileNetV2** | High | Fast | ~9 MB |
| **EfficientNetB0**| Very High| Moderate | ~16 MB |
| **ResNet50** | High | Slow | ~94 MB |

*Note: MobileNetV2 is the recommended architecture for deployment scenarios requiring low latency and minimal resource usage.*

## License

This project is open-sourced under the MIT License. See the [LICENSE](LICENSE) file for full details.

---
*Developed for research and educational purposes.*
