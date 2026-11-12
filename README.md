# Fusion Vision: Object Detection with Audio Transformation using Fourier Insights

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![Vision](https://img.shields.io/badge/MediaPipe-EfficientDet--Lite0-00A67E.svg)](https://developers.google.com/mediapipe)
[![Signal Processing](https://img.shields.io/badge/SciPy-Butterworth%20%26%20FFT-00599C.svg)](https://scipy.org/)

**Fusion Vision** is a multimodal AI desktop application that combines deep learning object detection (MediaPipe / EfficientDet-Lite0) with mathematical audio signal processing (Butterworth bandpass filtering & Fast Fourier Transform). It detects objects in real-time or from images, responds to voice commands, and gives spoken feedback.

---

## 🌟 Leadership & Team Member Roles

### 👑 Project Leader: **Yin Myat Noe Oo**
> **Huge Special Credit to Yin Myat Noe Oo (TNT - 1969)** for her exceptional leadership, vision, and technical direction. She orchestrated the end-to-end architecture, unified the computer vision and frequency-domain audio pipelines, resolved critical system integration hurdles, and steered **Team Pink (Rogue Five)** to a successful completion.

### 👥 Team Member Task Assignments
* **Yin Myat Noe Oo (TNT - 1969)** — *Project Leader & System Architect*  
  Overall project coordination, multimodal pipeline architecture, and integration of computer vision with audio processing.
* **Khin Sis Thway (TNT - 1979)** — *Computer Vision & Model Deployment*  
  Configured MediaPipe Object Detector with `efficientdet_lite0.tflite`, handled image frame transformations, and implemented dynamic bounding box overlays.
* **Cho May Moe (TNT - 1996)** — *Signal Processing & Fourier Analysis*  
  Implemented the 6th-order Butterworth bandpass filter ($300\text{ Hz} - 3000\text{ Hz}$), FFT frequency decomposition, and spectral visualization plots.
* **Hein Hlyan (TNT - 1998)** — *Audio Capture & Speech Recognition*  
  Implemented audio recording via `sounddevice` ($44.1\text{ kHz}$ PCM), speech-to-text processing with Google Speech API, and keyword validation.
* **Su Pyae Pyae Htut (TNT - 2051)** — *UI/UX & Text-to-Speech (TTS)*  
  Designed the Streamlit dashboard layout, custom CSS styling, user control panels, and `pyttsx3` audible voice feedback synthesis.

---

## 📐 Quick Mathematical Principles

1. **Butterworth Bandpass Filter ($300\text{ Hz} - 3000\text{ Hz}$):**  
   Isolates human vocal frequencies and removes ambient background noise with a maximally flat passband response:
   $$|H(j\omega)| = \frac{1}{\sqrt{1 + (\omega/\omega_c)^{2n}}} \quad (n = 6)$$

2. **Fast Fourier Transform (FFT):**  
   Converts discrete time-domain speech $x(n)$ to frequency components $X(k)$ in $\mathcal{O}(N \log N)$ complexity:
   $$X(k) = \sum_{n=0}^{N-1} x(n) e^{-j \frac{2\pi}{N} kn}, \quad f(k) = \frac{k \cdot f_s}{N}$$

---

## 🚀 Step-by-Step Setup & How to Run

### Step 1: Clone the Repository & Navigate to App Directory
```bash
git clone https://github.com/your-username/fusion-vision.git
cd fusion-vision/ObjectDetection_TeamPink
```

### Step 2: Create and Activate a Virtual Environment
* **Windows (Command Prompt / PowerShell):**
  ```powershell
  python -m venv .venv
  .venv\Scripts\activate
  ```
* **macOS / Linux:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

### Step 3: Install Required Dependencies
Run the following command to install the required libraries:
```bash
pip install streamlit opencv-python mediapipe numpy scipy sounddevice pyttsx3 SpeechRecognition matplotlib tensorflow
```

> **Windows Audio Tip:** If you encounter issues with `sounddevice` or `pyttsx3`, ensure your default microphone is enabled in Windows Settings and audio output drivers are active.

### Step 4: Launch the Streamlit App
Make sure you are in the `ObjectDetection_TeamPink` directory and run:
```bash
streamlit run detect8.py
```
This automatically opens your browser at `http://localhost:8501`.

---

## 🖥️ How to Use the Application

1. **Set Model Path:**  
   In the sidebar under **Model Path**, enter the relative or absolute path to the included model:
   ```text
   efficientdet_lite0.tflite
   ```
2. **Choose Input Source:**  
   * **Upload Image:** Browse and upload any `.jpg`, `.jpeg`, or `.png`. The model identifies objects and draws labeled green bounding boxes.
   * **Use Webcam:** Enables your live camera stream for continuous real-time detection.
3. **Voice Commands & Speech Feedback:**  
   * **🔊 Speak Detected Objects:** Click this button to make the system announce detected items out loud using text-to-speech.
   * **🎤 Listen for Voice Command:** Click to record your voice. Say a command containing words like `"detect"`, `"object"`, `"show"`, or `"classify"` to trigger automatic detection and speech output.
   * **📊 Analyze Voice Input Spectrum:** Visualizes the Fourier Transform graph (Frequency vs Amplitude) of your recorded audio command.
   * **🔄 Restart System:** Resets the application state.

---

## 📁 Project Structure

```text
fusion-vision/
├── README.md                          # Project documentation and quickstart
└── ObjectDetection_TeamPink/
    ├── detect8.py                     # Main application (Streamlit + Vision + Audio)
    ├── efficientdet_lite0.tflite      # Pre-trained TFLite object detection model
    ├── object_classification.h5       # Deep learning weights file
    └── detected_images/               # Test image samples
```
