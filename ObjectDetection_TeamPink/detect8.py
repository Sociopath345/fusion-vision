import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import scipy.fftpack as fft
import sounddevice as sd
import pyttsx3
import scipy.signal as signal
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from utils import visualize
import speech_recognition as sr
import os
import matplotlib.pyplot as plt

# Define keywords that must be present in the detected voice command
REQUIRED_KEYWORDS = ["detect", "object", "show", "please", "print", "classify", "output"]

# Custom CSS for luxurious styling
def custom_css():
    st.markdown("""
    <style>
        /* Change the sidebar (navigation bar) background color to a luxurious dark navy blue */
        [data-testid="stSidebar"] {
            background-color: #0A1F3D !important;  /* Deep navy blue */
        }

        /* Change the main content background color to a soft gold or cream */
        .stApp {
            background-color: #F5F5DC !important;  /* Soft cream */
        }

        /* Optional: Add a luxurious font and text color for better contrast in the sidebar */
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] p {
            color: #FFD700 !important;  /* Gold text for luxury feel */
            font-family: 'Georgia', serif;
        }

        /* Style buttons to have a red background with white text */
        [data-testid="stSidebar"] button {
            background-color: #8B0000 !important;  /* Dark red */
            color: white !important;
            border-radius: 5px;
            border: none;
            padding: 10px 20px;
            font-size: 16px;
            font-weight: bold;
        }

        /* Hover effect for buttons */
        [data-testid="stSidebar"] button:hover {
            background-color: #FF4500 !important;  /* Brighter red on hover */
            color: white !important;
        }
    </style>
    """, unsafe_allow_html=True)

def load_detector(model_path):
    if not os.path.exists(model_path):
        st.error("❌ Model file not found. Please provide a valid path.")
        return None
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.ObjectDetectorOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        score_threshold=0.5,
    )
    return vision.ObjectDetector.create_from_options(options)

def detect_objects(image, detector):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    results = detector.detect(mp_image)
    return results

def apply_fourier_transform(audio_signal, sample_rate):
    N = len(audio_signal)
    T = 1.0 / sample_rate
    frequencies = np.fft.fftfreq(N, T)
    transformed_signal = np.fft.fft(audio_signal)
    return transformed_signal, frequencies

def filter_background_noise(audio_signal, sample_rate):
    # Apply a bandpass filter to remove noise
    low_cutoff = 300.0  # Hz
    high_cutoff = 3000.0  # Hz
    sos = signal.butter(6, [low_cutoff, high_cutoff], btype='band', fs=sample_rate, output='sos')
    filtered_signal = signal.sosfilt(sos, audio_signal)
    return filtered_signal

def detect_voice_command(duration=5):
    st.write("🎤 Listening for a voice command...")
    sample_rate = 44100  # Hz
    audio_signal = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    # Convert the recorded audio signal to a format compatible with speech_recognition
    audio_signal_int16 = (audio_signal * 32767).astype(np.int16)  # Convert float32 to int16
    # Initialize the speech recognizer
    r = sr.Recognizer()
    audio_data = sr.AudioData(audio_signal_int16.tobytes(), sample_rate, 2)  # 2 bytes per sample for int16
    try:
        # Use Google Web Speech API to recognize the speech
        detected_text = r.recognize_google(audio_data)
        st.write(f"🎙️ Detected Text: {detected_text}")
        # Check if at least one required keyword is present in the detected text
        if any(keyword in detected_text.lower() for keyword in REQUIRED_KEYWORDS):
            st.write("✅ Valid voice command detected! Triggering speech output...")
            return True, audio_signal.flatten(), sample_rate  # Return audio signal and sample rate for Fourier Transform
        else:
            st.write("❌ Command not recognized. Please try again.")
            return False, None, None
    except sr.UnknownValueError:
        st.write("❌ Could not understand the audio. Please try again.")
        return False, None, None
    except sr.RequestError as e:
        st.write(f"❌ Error with the speech recognition service: {e}")
        return False, None, None

def generate_speech_signal(objects):
    if not objects:
        return  # Don't speak if no objects are detected
    # Remove duplicates by converting the list to a set, then back to a list
    unique_objects = list(set(objects))
    if len(unique_objects) == 1:
        full_text = f"🔊 The detected object is {unique_objects[0]}."
    else:
        full_text = f"🔊 The detected objects are {' and '.join(unique_objects)}."
    # Increase the font size of the printed output text
    st.markdown(f"<h3 style='text-align: center;'>{full_text}</h3>", unsafe_allow_html=True)
    engine = pyttsx3.init()
    engine.setProperty('rate', 125)  # Slow down speech rate
    engine.say(full_text)
    engine.runAndWait()

def restart_system():
    st.rerun()

def draw_detections(image, results, speak=False):
    detected_objects = []
    if results.detections:
        for detection in results.detections:
            bbox = detection.bounding_box
            start_point = (bbox.origin_x, bbox.origin_y)
            end_point = (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height)
            cv2.rectangle(image, start_point, end_point, (0, 255, 0), 2)
            label = detection.categories[0].category_name
            detected_objects.append(label)
            font_scale = max(0.5, min(1.5, image.shape[1] / 800))
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
            text_bg_start = (bbox.origin_x, max(0, bbox.origin_y - text_height - 5))
            text_bg_end = (bbox.origin_x + text_width + 10, bbox.origin_y)
            cv2.rectangle(image, text_bg_start, text_bg_end, (0, 255, 0), -1)
            cv2.putText(image, label, (bbox.origin_x + 5, bbox.origin_y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2)
    if speak and detected_objects:
        generate_speech_signal(detected_objects)
    return image, detected_objects

def run_webcam(detector, speak_button, duration):
    cap = cv2.VideoCapture(0)
    placeholder = st.empty()
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("❌ Failed to capture video from webcam.")
            break
        results = detect_objects(frame, detector)
        output_image, detected_objects = draw_detections(frame, results, speak=speak_button)
        placeholder.image(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB), channels="RGB")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def show_fourier_analysis(audio_signal, sample_rate):
    """Display Fourier Transform analysis of the audio signal."""
    if audio_signal is None or sample_rate is None:
        st.error("❌ No audio signal available for Fourier Transform analysis.")
        return
    transformed_signal, frequencies = apply_fourier_transform(audio_signal, sample_rate)
    # Plot the Fourier Transform
    plt.figure(figsize=(10, 4))
    plt.plot(frequencies[:len(frequencies) // 2], np.abs(transformed_signal)[:len(transformed_signal) // 2])
    plt.title("Fourier Transform of Voice Input")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.grid()
    st.pyplot(plt)

def main():
    # Inject custom CSS for luxurious styling
    custom_css()

    st.title("🚀 Real-Time Object Classification with Streamlit")
    model_path = st.sidebar.text_input("Model Path",
                                       "/Users/nico278/PycharmProjects/objectDetection/.venv/lib/efficientdet_lite0.tflite")
    detector = load_detector(model_path)
    if detector is None:
        return
    option = st.sidebar.radio("Select Input Source", ("Upload Image", "Use Webcam"))
    speak_button = st.sidebar.button("🔊 Speak Detected Objects")
    listen_button = st.sidebar.button("🎤 Listen for Voice Command")
    fourier_button = st.sidebar.button("📊 Analyze Voice Input Spectrum")  # New button for Fourier Transform
    duration = st.sidebar.slider("🎤 Recording Duration (sec)", 3, 10, 5)
    restart_button = st.sidebar.button("🔄 Restart System")
    if restart_button:
        restart_system()
    # Initialize audio_signal and sample_rate as session state variables
    if 'audio_signal' not in st.session_state:
        st.session_state.audio_signal = None
    if 'sample_rate' not in st.session_state:
        st.session_state.sample_rate = None
    if listen_button:
        success, st.session_state.audio_signal, st.session_state.sample_rate = detect_voice_command(duration)
        if success:
            speak_button = True
    if fourier_button:
        show_fourier_analysis(st.session_state.audio_signal, st.session_state.sample_rate)
    if option == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            results = detect_objects(image, detector)
            output_image, detected_objects = draw_detections(image, results)
            st.image(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB), channels="RGB")
            if speak_button and detected_objects:
                generate_speech_signal(detected_objects)
    elif option == "Use Webcam":
        run_webcam(detector, speak_button, duration)

if __name__ == "__main__":
    main()