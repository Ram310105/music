import streamlit as st
import cv2
import numpy as np
from emotion_processor import EmotionProcessor  # Import the EmotionProcessor class
import webbrowser

# Load the model and labels
model = EmotionProcessor("model.h5", "labels.npy")

# Streamlit app title
st.title("Emotion Based Music Recommender")

# Display the header for the application
st.header("Emotion Based Music Recommender")

# Initialize the emotion variable
emotion = ""

# Video capture function
def video_stream():
    global emotion  # Use global to modify the emotion variable directly
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame for emotion detection
        frame = model.recv(frame)

        # Convert BGR frame to RGB for Streamlit display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the frame in Streamlit
        st.image(frame_rgb)

        # Load the predicted emotion
        try:
            emotion = np.load("emotion.npy")[0]  # Load emotion prediction from saved file
        except:
            emotion = "Emotion Not Detected"

        if emotion != "Emotion Not Detected":
            st.write(f"Detected Emotion: {emotion}")

    cap.release()

# Start the video stream for real-time emotion detection
video_stream()

# Button to trigger the music recommendation based on emotion
btn = st.button("Recommend me songs")

if btn:
    if emotion == "":
        st.warning("Please let me capture your emotion first")
    else:
        st.write(f"Searching for {emotion} songs on YouTube...")
        webbrowser.open(f"https://www.youtube.com/results?search_query={emotion}+song")
        np.save("emotion.npy", np.array([""]))  # Reset emotion after recommendation
