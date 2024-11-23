import streamlit as st
import cv2
import mediapipe as mp
import math
import numpy as np
from PIL import Image

# Repository URL
REPO_URL = "https://github.com/Varun128-cell/Umar69"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def recognize_fingers(hand_landmarks):
    """
    Counts the number of extended fingers for one hand.
    """
    fingertips = [4, 8, 12, 16, 20]  # Thumb and finger tips
    finger_states = []  # 1 if finger is extended, 0 otherwise

    for i, tip in enumerate(fingertips):
        lower_joint = tip - 2 if i == 0 else tip - 3
        # Check if the tip is above the lower joint (indicating extension)
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[lower_joint].y:
            finger_states.append(1)
        else:
            finger_states.append(0)
    
    return sum(finger_states), finger_states  # Total extended fingers and their states

def recognize_gesture(finger_states):
    """
    Recognizes the gesture based on the states of the fingers.
    """
    if finger_states == [0, 0, 0, 0, 0]:
        return "Fist"
    elif finger_states == [1, 1, 1, 1, 1]:
        return "Open Hand"
    elif finger_states == [1, 0, 0, 0, 0]:
        return "Thumbs Up"
    elif finger_states == [0, 1, 1, 0, 0]:
        return "Peace"
    else:
        return "Unknown"

def recognize_heart_gesture(results, width, height):
    if len(results.multi_hand_landmarks) == 2:
        hand1 = results.multi_hand_landmarks[0]
        hand2 = results.multi_hand_landmarks[1]
        x1, y1 = (hand1.landmark[8].x * width, hand1.landmark[8].y * height)
        x2, y2 = (hand2.landmark[8].x * width, hand2.landmark[8].y * height)
        distance = calculate_distance(x1, y1, x2, y2)
        slope = (y2 - y1) / (x2 - x1 + 1e-5)
        relative_threshold = width * 0.2
        if distance < relative_threshold and -1.5 < slope < -0.3:
            return True, (int(x1), int(y1)), (int(x2), int(y2))
    return False, None, None

def process_frame(frame):
    """
    Processes each frame to count total fingers and recognize gestures.
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    h, w, _ = frame.shape
    total_fingers = 0
    gestures = []
    heart_detected = False

    if results.multi_hand_landmarks:
        heart_detected, point1, point2 = recognize_heart_gesture(results, w, h)
        for hand_landmarks in results.multi_hand_landmarks:
            # Count fingers for each detected hand
            fingers, finger_states = recognize_fingers(hand_landmarks)
            total_fingers += fingers
            gesture = recognize_gesture(finger_states)
            gestures.append(gesture)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        if heart_detected:
            gestures = ["Heart"]  # Override gestures if heart is detected
            cv2.line(frame, point1, point2, (0, 255, 0), 2)
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame, total_fingers, gestures

def registration():
    st.markdown("""
        <style>
        body {
            background: linear-gradient(90deg, #ff9a9e, #fad0c4);
            color: #333333;
        }
        h1 {
            font-family: 'Arial Black', sans-serif;
            color: #ffffff;
            text-align: center;
            text-shadow: 2px 2px 4px #333;
        }
        </style>
    """, unsafe_allow_html=True)
    st.markdown("<h1>🤝 Welcome to Gesture Recognition</h1>", unsafe_allow_html=True)
    st.write("Please register to continue!")
    name = st.text_input("Enter your Name")
    email = st.text_input("Enter your Email")
    if st.button("Register"):
        if name and email:
            st.session_state.registered = True
            st.session_state.name = name
            st.session_state.email = email
            st.success(f"Welcome, {name}!")
            return True
        else:
            st.error("Please fill in all fields.")
    return False

def main():
    if 'registered' not in st.session_state:
        st.session_state.registered = False

    if not st.session_state.registered:
        registered = registration()
        if not registered:
            return
    else:
        st.set_page_config(page_title="Gesture Recognition", layout="wide")
        st.markdown("""
            <style>
            body {
                background: linear-gradient(45deg, #ffe259, #ffa751);
                color: #333333;
            }
            h1 {
                font-family: 'Arial Black', sans-serif;
                color: #ffffff;
                text-align: center;
                text-shadow: 2px 2px 4px #333;
            }
            .sidebar {
                background-color: #ffecb3;
                padding: 15px;
                border-radius: 15px;
            }
            </style>
        """, unsafe_allow_html=True)
        st.markdown("<h1>🤖 Gesture Recognition</h1>", unsafe_allow_html=True)
        st.write("Use your webcam or upload a photo to recognize hand gestures!")
        st.sidebar.markdown("<div class='sidebar'><h2>Options</h2></div>", unsafe_allow_html=True)
        st.sidebar.write(f"[GitHub Repository]({REPO_URL})")
        mode = st.sidebar.radio("Select Input Mode:", ["Upload Photo", "Live Camera Capture"])
        st.sidebar.markdown("---")
        st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/a/a3/Logo-placeholder.svg/1200px-Logo-placeholder.svg.png", width=150)
        st.sidebar.markdown("""
            <div style="text-align: center; margin-top: 20px;">
                <h4>Developed by</h4>
                <h3 style="color: #ff6f61;">Varun</h3>
            </div>
        """, unsafe_allow_html=True)

        if mode == "Upload Photo":
            uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
            if uploaded_file:
                image = Image.open(uploaded_file)
                frame = np.array(image)
                processed_frame, total_fingers, gestures = process_frame(frame)
                st.image(processed_frame, caption=f"Total Fingers: {total_fingers} | Gestures: {', '.join(gestures)}", use_column_width=True)

        elif mode == "Live Camera Capture":
            FRAME_WINDOW = st.image([])
            gesture_display = st.empty()
            run = st.sidebar.checkbox("Start Camera")
            cap = cv2.VideoCapture(0)

            if run:
                while run:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to access the camera.")
                        break
                    frame = cv2.flip(frame, 1)
                    processed_frame, total_fingers, gestures = process_frame(frame)
                    FRAME_WINDOW.image(processed_frame, channels="RGB")
                    gesture_display.markdown(f"### Total Fingers: **{total_fingers}** | Gestures: **{', '.join(gestures)}**", unsafe_allow_html=True)

                cap.release()
            else:
                gesture_display.markdown("### Total Fingers: 0 | Gestures: None", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
